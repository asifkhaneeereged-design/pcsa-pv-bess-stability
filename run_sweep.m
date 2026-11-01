%% run_sweep.m  --  full dataset: 7 SCR levels x 2 controllers x 34 disturbance cases = 476 runs
% Calls run_scenario.m for every configuration and skips runs whose CSV already exists,
% so an interrupted sweep resumes where it stopped when run_sweep is called again.
% Output: folder given by the environment variable PCSA_DATA (default ./data)
%           runs\<run_id>.csv   +   run_log.csv

N_LIMIT = inf;      % set to a small number for a test; inf for the full sweep

OUTDIR = getenv('PCSA_DATA'); if isempty(OUTDIR), OUTDIR = fullfile(pwd, 'data'); end
ctrls  = {'GFL', 'VSM'};
levels = {'G1','G2','G3','G4','G5','G6','G7'};   % SCR 0.44 ... 3.52

% ---------------- 34 disturbance cases ----------------
% {scenario, tag, overrides}
C = {};
C(end+1,:) = {0, 'steady', {}};                                         % 1
for f = [0.9 0.7 0.5 0.3]                                               % 4
    C(end+1,:) = {1, sprintf('pv%02d', round(100*(1-f))), {'PV_after_factor', f}};
end
for P = [10 20 30 40 60]                                                % 5
    C(end+1,:) = {2, sprintf('lr%02d', P), {'Load.P2', P*1e6}};
end
for P = [20 40 60]                                                      % 3
    C(end+1,:) = {3, sprintf('isl%02d', P), {'Load.P1', P*1e6}};
end
for d = [0.05 0.10 0.15 0.20 0.30 0.50]                                 % 18
    for x = [0.10 0.50 0.99]
        C(end+1,:) = {4, sprintf('tf_d%03d_x%02d', round(1000*d), round(100*x)), ...
                      {'Scenario.Fault_duration', d; 'Scenario.Fault_distance', x}};
    end
end
for x = [0.10 0.50 0.99]                                                % 3
    C(end+1,:) = {5, sprintf('pf_x%02d', round(100*x)), {'Scenario.Fault_distance', x}};
end

% ---------------- run list ----------------
% Order: case -> controller -> SCR, so that the completed cases cover
% all conditions even if the sweep is interrupted.
jobs = {};
for c = 1:size(C,1)
    for k = 1:numel(ctrls)
        for g = 1:numel(levels)
            id = sprintf('S%d_%s_%s_%s', C{c,1}, ctrls{k}, levels{g}, C{c,2});
            jobs(end+1,:) = {id, C{c,1}, ctrls{k}, levels{g}, C{c,3}}; %#ok<SAGROW>
        end
    end
end
N = min(size(jobs,1), N_LIMIT);
fprintf('Total %d runs (disturbances %d x controllers %d x SCR %d). To run now: %d\n', ...
        size(jobs,1), size(C,1), numel(ctrls), numel(levels), N);

% ---------------- run ----------------
t0 = tic; nDone = 0; nErr = 0;
for i = 1:N
    csvPath = fullfile(OUTDIR, 'runs', [jobs{i,1} '.csv']);
    if exist(csvPath, 'file'), continue; end
    cfg = struct('run_id', jobs{i,1}, 'scenario', jobs{i,2}, 'ctrl', jobs{i,3}, ...
                 'grid', jobs{i,4}, 'outdir', OUTDIR);
    cfg.overrides = jobs{i,5};
    fprintf('\n(%d/%d) ', i, N);
    try
        info = run_scenario(cfg);
        if startsWith(string(info.status), "error"), nErr = nErr + 1; end
    catch e
        nErr = nErr + 1;
        fprintf(2, '[crash] %s: %s\n', jobs{i,1}, e.message);
    end
    nDone = nDone + 1;
    el = toc(t0); left = (N - i) * el / nDone;
    fprintf('   elapsed %.1f min | baki ~%.1f min | error %d\n', el/60, left/60, nErr);
end

fprintf('\n===== DONE: %d notun run, %d error, mot %.1f min =====\n', nDone, nErr, toc(t0)/60);
fprintf('Log: %s\n', fullfile(OUTDIR, 'run_log.csv'));
