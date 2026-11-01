function info = run_scenario(cfg)
%RUN_SCENARIO  Runs one configured simulation and saves the waveform CSV and a run-log entry.
%
% cfg fields (all optional):
%   run_id    : unique name, e.g. 'S4_GFL_G4_tf_d150_x50'
%   scenario  : 0 = steady state, 1 = PV variation, 2 = load switching,
%               3 = grid outage, 4 = temporary fault, 5 = permanent fault
%   ctrl      : 'GFL' | 'VSM' | 'Droop'
%   grid      : 'weak' | 'medium' | 'strong'
%   t_event   : disturbance time (s), default 4
%   Tsim      : simulation time (s), default 6
%   overrides : {'Variable.name', value; ...} -- applied after the scenario settings
%   outdir    : output folder
%
% Example:
%   cfg = struct('run_id','TEST_fault_GFL','scenario',4,'ctrl','GFL','grid','weak');
%   run_scenario(cfg)

mdl   = 'BatteryStoragePVPlantGFM';
scope = [mdl '/Scope/Scope'];
gfm   = [mdl '/BESS PV PARK/60 MWhr Battery Energy Storage/Variant Subsystem/' ...
         'Grid Forming VSM/GridFormingController'];

% ---------------- defaults ----------------
def.run_id = ''; def.scenario = 0; def.ctrl = 'GFL'; def.grid = 'weak';
def.t_event = 4; def.Tsim = 6; def.overrides = {};
def.outdir = getenv('PCSA_DATA'); if isempty(def.outdir), def.outdir = fullfile(pwd, 'data'); end
fn = fieldnames(def);
for k = 1:numel(fn)
    if ~isfield(cfg, fn{k}) || isempty(cfg.(fn{k})), cfg.(fn{k}) = def.(fn{k}); end
end
if isempty(cfg.run_id)
    cfg.run_id = sprintf('S%d_%s_%s_%s', cfg.scenario, cfg.ctrl, cfg.grid, datestr(now,'HHMMSS'));
end

runDir  = fullfile(cfg.outdir, 'runs');
if ~exist(runDir, 'dir'), mkdir(runDir); end
csvPath = fullfile(runDir, [cfg.run_id '.csv']);
info = cfg; info.status = "skipped"; info.sim_time_s = NaN;
if exist(csvPath, 'file')
    fprintf('[skip] %s (CSV already exists)\n', cfg.run_id); return;
end

open_system(mdl);
set_param(mdl, 'SimscapeLogType', 'none');     % disable large Simscape logs
set_param(mdl, 'InspectSignalLogs', 'off');    % do not stream signals to the Simulation Data Inspector
Simulink.sdi.clear;                            % clear previously stored data

% ---------------- 1) reset all parameters to defaults ----------------
assignin('base', 'SimulationTime', cfg.Tsim);
evalin('base', 'run(''BatteryStoragePVPlantGFMParameters'')');

% ---------------- 2) controller ----------------
switch upper(cfg.ctrl)
    case 'GFL',   bc = 2; dr = 0;
    case 'VSM',   bc = 1; dr = 0; set_param(gfm, 'powerControl', 'Virtual Synchronous Machine');
    case 'DROOP', bc = 1; dr = 1; set_param(gfm, 'powerControl', 'Droop Control');
    otherwise, error('ctrl hobe GFL, VSM ba Droop');
end
assignin('base', 'BESSControl', bc);
assignin('base', 'droop', dr);

% ---------------- 3) grid strength ----------------
% The GridStrength flag of the model changes no block. In the SCR calculation
% the grid-source impedance (Grid.Rs, Grid.Ls) dominates, followed by the MV feeder.
% Grid strength is therefore set by the source-impedance scale and the feeder length.
% PV penetration (Array.insolation) is kept fixed at 800 W/m2.
%   weak   : Z x 1.00, feeder2 0.60 km  -> SCR ~0.44 (model default)
%   medium : Z x 0.25, feeder2 0.10 km  -> SCR ~1.5
%   strong : Z x 0.05, feeder2 0.01 km  -> SCR ~3.5
%   G1..G7 : the seven SCR levels of the sweep (0.44 ... 3.52)
switch lower(cfg.grid)
    case {'weak','g1'},   zs = 1.00; f2 = 0.60;   % SCR ~0.44
    case 'g2',            zs = 0.80; f2 = 0.45;   % SCR ~0.54
    case 'g3',            zs = 0.60; f2 = 0.35;   % SCR ~0.69
    case 'g4',            zs = 0.45; f2 = 0.25;   % SCR ~0.89
    case 'g5',            zs = 0.35; f2 = 0.18;   % SCR ~1.11
    case {'medium','g6'}, zs = 0.25; f2 = 0.10;   % SCR ~1.47
    case {'strong','g7'}, zs = 0.05; f2 = 0.01;   % SCR ~3.52
    otherwise, error('grid hobe weak/medium/strong ba G1..G7');
end
assignin('base', 'GridStrength', double(zs < 1));
evalin('base', sprintf('Grid.Rs = Grid.Rs*%g; Grid.Ls = Grid.Ls*%g; Feeder.lengthSecondHalf = %g;', ...
                       zs, zs, f2));

% ---------------- 3b) PV drop magnitude ----------------
% The irradiance factor hard-coded in the PV step blocks is replaced by a variable
% PV_after_factor = 0.7 means a 30% irradiance drop (model default)
pvSteps = {[mdl '/BESS PV PARK/100 MWp PV Park/Step1'], ...
           [mdl '/BESS PV PARK/100 MWp PV Park/Step2']};
for k = 1:numel(pvSteps)
    set_param(pvSteps{k}, 'After', 'Array.insolation*PV_after_factor');
end
assignin('base', 'PV_after_factor', 0.7);

% ---------------- 4) scenario ----------------
assignin('base', 'Load2CB_initial', 0);
if cfg.scenario == 0
    late = 100 * cfg.Tsim;                      % move all events beyond the end of the simulation
    evalin('base', sprintf(['Scenario.Fault_time=%g; Scenario.Load_switching_time=%g; ' ...
        'Scenario.Solar_flactuation_time=%g; Scenario.Grid_outage_time=%g; ' ...
        'Scenario.Line_trip_time=%g;'], late, late, late, late, late));
    assignin('base', 'Load2CB_final', 0);
else
    evalin('base', sprintf('Scenario.number=%d; Scenario.t_event=%g;', cfg.scenario, cfg.t_event));
    assignin('base', 'Load2CB_final', double(cfg.scenario == 2));
    evalin('base', ['P_ = BatteryStoragePVPlantGFMSettingScenario(Scenario.number, SimulationTime, Scenario.t_event); ' ...
        'Scenario.Fault_distance=P_(1); Scenario.Fault_time=P_(2); Scenario.Fault_duration=P_(3); ' ...
        'Scenario.Load_switching_time=P_(4); Scenario.Solar_flactuation_time=P_(5); ' ...
        'Scenario.Grid_outage_time=P_(6); Scenario.Line_trip_time=P_(7);']);
end

% ---------------- 5) overrides (magnitude, H, D ...) ----------------
ov = cfg.overrides; ovStr = "";
for k = 1:size(ov, 1)
    evalin('base', sprintf('%s = %s;', ov{k,1}, mat2str(ov{k,2})));
    ovStr = ovStr + sprintf('%s=%s; ', ov{k,1}, mat2str(ov{k,2}));
end

% ---------------- 5b) SCR calculation (after the overrides) ----------------
evalin('base', 'PVpower = Array.insolation*2*50e6/1e3;');
try
    scrS = evalin('base', 'BatteryStoragePVPlantGFMSCRCal(Line,Feeder,Grid,SubTransmissionLine,PVpower)');
    SCR = scrS.SCR;
catch
    SCR = NaN;
end
feederKm  = evalin('base', 'Feeder.lengthSecondHalf');
insol     = evalin('base', 'Array.insolation');

% ---------------- 6) Scope logging ----------------
set_param(scope, 'DataLogging','on', 'DataLoggingVariableName','ScopeRun', ...
          'DataLoggingSaveFormat','Dataset', 'DataLoggingLimitDataPoints','off');

% ---------------- 7) simulate + CSV ----------------
names = { {'P_total_MW'}, {'Q_total_MVAr'}, {'Vmag_POI_pu'}, {'f_POI_Hz'}, ...
          {'Va_POI_pu','Vb_POI_pu','Vc_POI_pu'}, {'Ia_POI_pu','Ib_POI_pu','Ic_POI_pu'}, ...
          {'P_BESS_MW'}, {'Q_BESS_MVAr'}, {'P_PV_MW'}, {'Q_PV_MVAr'}, ...
          {'Ia_BESS_pu','Ib_BESS_pu','Ic_BESS_pu'} };
Ts_out = 2e-4;
try
    tic;
    out  = sim(mdl, 'StopTime', num2str(cfg.Tsim), 'ReturnWorkspaceOutputs', 'on');
    simT = toc;
    ds = out.get('ScopeRun');
    t  = (0:Ts_out:cfg.Tsim)';
    T  = table(repmat(string(cfg.run_id), numel(t), 1), t, 'VariableNames', {'run_id','Time_s'});
    for k = 1:numel(names)
        ts = ds.getElement(k).Values;
        x  = squeeze(ts.Data);
        if size(x,1) ~= numel(ts.Time), x = x.'; end
        [tu, iu] = unique(ts.Time);
        xi = interp1(tu, x(iu,:), t, 'linear', 'extrap');
        for c = 1:numel(names{k}), T.(names{k}{c}) = xi(:,c); end
    end
    writetable(T, csvPath);
    clear out ds;  Simulink.sdi.clear;         % clear temporary data after each run
    status = "ok";
catch e
    simT = NaN;
    status = "error: " + string(e.message);
    fprintf(2, '[error] %s: %s\n', cfg.run_id, e.message);
end

% ---------------- 8) run_log.csv ----------------
L = table(string(cfg.run_id), cfg.scenario, string(cfg.ctrl), string(cfg.grid), ...
          evalin('base','BESSControl'), evalin('base','droop'), zs, feederKm, insol, SCR, ...
          cfg.t_event, cfg.Tsim, ovStr, simT, status, string(datestr(now, 'yyyy-mm-dd HH:MM:SS')), ...
          'VariableNames', {'run_id','scenario','ctrl','grid','BESSControl','droop', ...
                            'gridZscale','feeder2_km','insolation','SCR', ...
                            't_event','Tsim','overrides','sim_time_s','status','timestamp'});
logPath = fullfile(cfg.outdir, 'run_log.csv');
if exist(logPath, 'file'), writetable(L, logPath, 'WriteMode', 'append');
else, writetable(L, logPath); end

info.status = status; info.sim_time_s = simT;
info.SCR = SCR;
fprintf('[%s] %s  (%.0f s)  SCR = %.2f\n', status, cfg.run_id, simT, SCR);
end
