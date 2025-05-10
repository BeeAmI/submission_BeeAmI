function [RMSE, MSE, n_predictions, MSE_per_timestep, correct_angle_per_timestep, correct_angle_predictions, angle_predictions, wrong_angle_indices, wrong_angle_MSE, time, train_time, val_time] = testScript(model, visualize, granular, seed, validation, folds)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Parts of the code generated with Github Copilot.
    % Run test script with cross-validation.
    % args:
    % model: (string) folder name of the model under /BeeAmI
    % visualize: (float) delay_for_plotting i.e. 0 for no plotting, 1000 for 1s
    % granular: (boolean) to plot each trial or not to plot, that is the question
    % seed: (int) chosen rng config
    % validation: (boolean) whether used for validation
    % folds: (int) number of cross-validation folds
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if not(validation)
        clc;
        close all;
    end
    warning('off', 'all');
    
    global is_paused;
    is_paused = false;

    load monkeydata_training.mat
    rng(seed);
    ix = randperm(length(trial));

    if not(exist('folds', 'var')) || folds < 2
        folds = 2;
    end

    addpath(['BeeAmI/', model]);
    angle_models = {'follower', 'gaming', 'lda_mean_trajectory', 'velocity', ...
     'velocity_units', 'velocity_angle', 'velocity_xyz', 'linear_trajectory', ...
     'wiener', 'kalman_diff', 'shifted_1', 'shifted_2', 'shifted_3', 'shifted_pca', ...
     'shifted_pca_fr_1', 'shifted_pca_fr_2', 'shifted_pca_fr_online', 'velocity_pca_fr', ...
     'shifted_pca_fr_1_online', 'shifted_pca_fr_binary', 'shifted_pca_fr_change', ...
     'shifted_pca_fr_change_arg', 'shifted_pca_fr_1_online_2', 'shifted_velocity_1.1', ...
     'shifted_velocity_1.2', 'velocity_accel', 'surprise', 'follower_pca_fr_1', ...
     'shifted_online_fast_1.1', 'follower_pca_fr_2', 'shifted_pca_fr_online_angles', ...
     'shifted_fast_follower', 'shifted_velocity_1.3', 'shifted_velocity_1.4', 'shifted_velocity_1.5', ...
     'shifted_velocity_1.6', 'shifted_velocity_1.7', 'vfinal_1', 'vfinal_2', 'vfinal_3', 'vfinal_4', ...
     'vfinal_5', 'vfinal_6', 'shifted_online_fast_1.2', 'shifted_online_fast_1.3', 'shifted_velocity_1.8', ...
     'shifted_velocity_1.9', 'shifted_online_fast_1.4', 'shifted_online_fast_1.5', 'shifted_online_fast_1.6', ...
     'shifted_online_fast_1.7', 'shifted_velocity_2.1', 'shifted_velocity_3.1', 'shifted_velocity_3.2', ...
     'x_final', 'x_final_avg', 'x_final_avg', 'x_final_lda', 'x_final_lda_pair', 'x_final_prob', 'x_final_vel', ...
     'x_final_vel_avg', 'x_final_start', 'avg_fast', 'fc_online', 'fc_template', 'lda_online', 'lda_template', 'z_1'};
    qda_models = {'shifted_online_fast_1.6'};
    
    % Initialize arrays for storing fold-specific metrics
    fold_RMSE = zeros(folds, 1);
    fold_MSE = zeros(folds, 1);
    fold_n_predictions = zeros(folds, 1);
    fold_correct_angle_predictions = zeros(folds, 1);
    fold_angle_predictions = zeros(folds, 1);
    fold_wrong_angle_MSE = zeros(folds, 1);
    fold_wrong_angle_indices = zeros(8, folds);
    fold_train_time = zeros(folds, 1);
    fold_val_time = zeros(folds, 1);
    
    % Determine the maximum timestep size for all trials
    max_timestep_size = 0;
    for i = 1:size(trial, 1)
        for j = 1:size(trial, 2)
            current_size = size(trial(i,j).spikes, 2);
            if current_size > max_timestep_size
                max_timestep_size = current_size;
            end
        end
    end
    
    first_timestep = 320;
    timesteps = floor(max_timestep_size/20);
    
    % Initialize aggregate MSE per timestep and correct angle per timestep
    agg_MSE_per_timestep = cell(timesteps, 1);
    for t = 1:timesteps
        agg_MSE_per_timestep{t} = [];
    end
    
    agg_correct_angle_per_timestep = cell(timesteps, 1);
    for t = 1:timesteps
        agg_correct_angle_per_timestep{t} = [];
    end
    
    %% Cross-validation loop
    fprintf('Starting %d-fold cross-validation for model: %s\n', folds, model);
    total_time_start = tic;
    
    % For CV, we need to divide data into folds
    fold_size = floor(length(trial) / folds);
    
    for fold = 1:folds
        fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nRunning Fold %d/%d\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n', fold, folds);
        
        % Determine validation indices for this fold
        val_start = (fold - 1) * fold_size + 1;
        val_end = fold * fold_size;
        if fold == folds
            val_end = length(trial);
        end
        
        val_indices = ix(val_start:val_end);
        train_indices = setdiff(ix, val_indices);
        
        % Create training and validation sets
        trainingData = trial(train_indices, :);
        valData = trial(val_indices, :);
        num_trials = size(valData, 1);
        
        % Initialize fold-specific metrics
        MSE = 0;
        MSE_per_timestep = cell(timesteps, 1);
        for t = 1:timesteps
            MSE_per_timestep{t} = [];
        end
        
        n_predictions = 0;
        angle_predictions = 0;
        correct_angle_predictions = 0;
        wrong_angle_MSE = 0;
        wrong_angle_indices = zeros(8, 1);
        
        % Initialize storage for bad trajectories within this fold
        bad_trajectories = struct('trial', {}, 'direction', {}, 'decodedX', {}, 'decodedY', {}, 'actualX', {}, 'actualY', {}, 'MSE', {});
        bad_angle_trajectories = struct('trial', {}, 'direction', {}, 'decodedX', {}, 'decodedY', {}, 'actualX', {}, 'actualY', {}, 'MSE', {});
        correct_angle_per_timestep = cell(timesteps, 1);
        for t = 1:timesteps
            correct_angle_per_timestep{t} = [];
        end

        %% Train Model for this fold
        fprintf('Training model for fold %d...\n', fold);
        fold_start_time = tic;
        modelParameters = positionEstimatorTraining(trainingData);
        if visualize && fold == 1  % Only visualize for first fold
            if ismember(qda_models, model)
                plotQDABoundaries(modelParameters, trainingData);
            else
                plotLDABoundaries(modelParameters, trainingData);
            end
        end
        train_time = toc(fold_start_time);
        fprintf('Training completed in %.2f seconds\n', train_time);

        %% Initiliaze plots for this fold
        if visualize && fold == 1  % Only visualize for first fold
            fig = figure('Position', [400 50 1200 950]);
            uipanel('Position', [0 0 1 0.08], 'Parent', fig);
            pause_btn = uicontrol('Style', 'togglebutton', 'String', 'Pause', ...
                'Position', [450 10 100 30], 'Parent', fig, ...
                'Callback', @toggle_pause);
            
            ax1 = subplot('Position', [0.05 0.15 0.25 0.7]);
            hold(ax1, 'on');
            axis(ax1, 'square');
            axis(ax1, [-150 150 -150 150]);
            grid(ax1, 'on');
            title(ax1, sprintf('All Trials - Fold %d/%d', fold, folds));
            subtitle(ax1, 'Decoded - red, actual - blue');
            
            if granular
                ax2 = subplot('Position', [0.38 0.15 0.25 0.7]);
                hold(ax2, 'on');
                axis(ax2, 'square');
                grid(ax2, 'on');
                title(ax2, 'Current Trial');

                ax3 = subplot('Position', [0.71 0.15 0.25 0.7]);
                hold(ax3, 'on');
                axis(ax3, 'square');
                grid(ax3, 'on');
                title(ax3, 'Current Trial');
            end
        end

        %% Test for this fold
        fprintf('Testing fold %d...\n', fold);
        for tr=1:num_trials
            if not(validation)
                display(['Decoding block ',num2str(tr),' out of ',num2str(num_trials), ' (Fold ', num2str(fold), ')']);
            end
            
            if visualize && granular && fold == 1  % Only visualize for first fold
                subplot(ax2)
                cla
                hold on
                axis square
                grid

                subplot(ax3)
                cla
                hold on
                axis square
                grid
            end

            if ismember(model, angle_models)
                correct_angle_predictions_t = 0;
                angle_predictions_t = 0;
                wrong_angle_indices_t = zeros(8, 1); 
            end
            
            %% Direction
            for direc=randperm(8) 
                decodedHandPos = [];
                times=first_timestep:20:size(valData(tr,direc).spikes,2);
                mse_per_angle = 0;

                %% Timesteps
                for t=times
                    if visualize && fold == 1  % Only check pause for first fold visualization
                        check_pause();
                    end
                    
                    past_current_trial.trialId = valData(tr,direc).trialId;
                    past_current_trial.spikes = valData(tr,direc).spikes(:,1:t);
                    past_current_trial.decodedHandPos = decodedHandPos;
                    past_current_trial.startHandPos = valData(tr,direc).handPos(1:2,1); 
                    
                    %% Estimator outputs
                    if nargout('positionEstimator') == 4 && ismember(model, angle_models)
                        [decodedPosX, decodedPosY, newParameters, angle] = positionEstimator(past_current_trial, modelParameters);
                        modelParameters = newParameters;
                    elseif nargout('positionEstimator') == 3 && ismember(model, angle_models)
                        [decodedPosX, decodedPosY, angle] = positionEstimator(past_current_trial, modelParameters);
                    elseif nargout('positionEstimator') == 3
                        [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                        modelParameters = newParameters;
                    elseif nargout('positionEstimator') == 2
                        [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
                    end
                    
                    decodedPos = [decodedPosX; decodedPosY];
                    decodedHandPos = [decodedHandPos decodedPos];
                    
                    mse = norm(valData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                    MSE = MSE + mse;
                    % (t/20 - 15) to get indexes from 1 to the last timestep
                    t_idx = (t/20 - 15);
                    MSE_per_timestep{t_idx} = [MSE_per_timestep{t_idx}, mse];
                    mse_per_angle = mse_per_angle + mse;
                    
                    if ismember(model, angle_models)
                        correct_angle = direc == angle;
                        correct_angle_per_timestep{t_idx} = [correct_angle_per_timestep{t_idx}, correct_angle];
                        if not(correct_angle)
                            wrong_angle_MSE = wrong_angle_MSE + mse;
                        end
                        if t == first_timestep
                            if correct_angle
                                correct_angle_predictions_t = correct_angle_predictions_t + 1;
                            else
                                wrong_angle_indices_t(direc) = wrong_angle_indices_t(direc) + 1;
                            end
                            angle_predictions_t = angle_predictions_t + 1;
                        end
                    end

                    %% Granular timestep plots (only for first fold)
                    if visualize && granular && fold == 1
                        subplot(ax3)
                        cla
                        hold on
                        plot(valData(tr,direc).handPos(1,1:t), valData(tr,direc).handPos(2,1:t), 'b-')
                        plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r-')
                        plot(decodedPosX, decodedPosY, 'ro', 'MarkerFaceColor', 'r')
                        plot(valData(tr,direc).handPos(1,t), valData(tr,direc).handPos(2,t), 'bo', 'MarkerFaceColor', 'b')
                        
                        title(sprintf('Fold: %d, Trial: %d, Timestep: %d, Direction: %d, MSE: %.2f', fold, tr, t, direc, mse))
                        if ismember(model, angle_models) && correct_angle
                            subtitle(sprintf('Angle prediction: %s', 'Correct'))
                        else
                            subtitle(sprintf('Angle prediction: %s', 'Incorrect'))
                        end
                        drawnow
                        pause(visualize/1000)
                    end
                end
                n_predictions = n_predictions + length(times);
                mse_per_angle = mse_per_angle / length(times);

                %% Add bad trajectories (only track for first fold)
                if fold == 1 && ismember(model, angle_models)
                    MSE_threshold = 200;
                    if (mse_per_angle > MSE_threshold) && correct_angle
                        bad_trajectories(end+1) = struct(...
                            'trial', tr, ...
                            'direction', direc, ...
                            'decodedX', decodedHandPos(1,:), ...
                            'decodedY', decodedHandPos(2,:), ...
                            'actualX', valData(tr,direc).handPos(1,times), ...
                            'actualY', valData(tr,direc).handPos(2,times), ...
                            'MSE', mse_per_angle ...
                        );
                    elseif not(correct_angle)
                        bad_angle_trajectories(end+1) = struct(...
                            'trial', tr, ...
                            'direction', direc, ...
                            'decodedX', decodedHandPos(1,:), ...
                            'decodedY', decodedHandPos(2,:), ...
                            'actualX', valData(tr,direc).handPos(1,times), ...
                            'actualY', valData(tr,direc).handPos(2,times), ...
                            'MSE', mse_per_angle ...
                        );
                    end
                end

                %% Main plot (only for first fold)
                if visualize && fold == 1
                    subplot(ax1)
                    hold on
                    plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r')
                    plot(valData(tr,direc).handPos(1,times),valData(tr,direc).handPos(2,times),'b')
                    drawnow
                end

                %% Granular trial plots (only for first fold)
                if visualize && granular && fold == 1
                    subplot(ax2)
                    hold on
                    plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r')
                    plot(valData(tr,direc).handPos(1,times), valData(tr,direc).handPos(2,times), 'b')
                    title(sprintf('Fold: %d, Trial: %d, Last Direction: %d, MSE: %.2f', fold, tr, direc, mse_per_angle))

                    if ismember(model, angle_models)
                        subtitle(sprintf('Angle predictions: %d / %d', correct_angle_predictions_t, angle_predictions_t))
                    end
                end

                if ismember(model, angle_models)
                    correct_angle_predictions = correct_angle_predictions + correct_angle_predictions_t;
                    angle_predictions = angle_predictions + angle_predictions_t;
                    wrong_angle_indices = wrong_angle_indices + wrong_angle_indices_t;
                end
            end
        end
        
        % Calculate fold-specific metrics
        fold_train_time(fold) = train_time;
        fold_val_time(fold) = toc(fold_start_time) - train_time;
        fold_RMSE(fold) = sqrt(MSE/n_predictions);
        fold_MSE(fold) = MSE;
        fold_n_predictions(fold) = n_predictions;
        
        if ismember(model, angle_models)
            fold_correct_angle_predictions(fold) = correct_angle_predictions;
            fold_angle_predictions(fold) = angle_predictions;
            fold_wrong_angle_MSE(fold) = wrong_angle_MSE;
            fold_wrong_angle_indices(:, fold) = wrong_angle_indices;
        end
        
        % Aggregate the MSE per timestep and correct angle per timestep
        for t = 1:timesteps
            if ~isempty(MSE_per_timestep{t})
                agg_MSE_per_timestep{t} = [agg_MSE_per_timestep{t}, MSE_per_timestep{t}];
            end
            
            if ismember(model, angle_models) && ~isempty(correct_angle_per_timestep{t})
                agg_correct_angle_per_timestep{t} = [agg_correct_angle_per_timestep{t}, correct_angle_per_timestep{t}];
            end
        end
        
        % Report fold results
        fprintf('\nResults for Fold %d:\n', fold);
        fprintf('RMSE: %.2f\n', fold_RMSE(fold));
        fprintf('Total MSE: %.2f\n', fold_MSE(fold));
        fprintf('Number of predictions: %d\n', fold_n_predictions(fold));
        
        if ismember(model, angle_models)
            fold_correct_angle_prcnt = fold_correct_angle_predictions(fold)/fold_angle_predictions(fold)*100;
            fold_wrong_angle_MSE_prcnt = (fold_wrong_angle_MSE(fold)/fold_MSE(fold)*100);
            fprintf('Correct angle predictions: %d\n', fold_correct_angle_predictions(fold));
            fprintf('Total angle predictions: %d\n', fold_angle_predictions(fold));
            fprintf('Correct angle predictions percentage: %.2f%%\n', fold_correct_angle_prcnt);
            fprintf('Wrong angle MSE: %.2f\n', fold_wrong_angle_MSE(fold));
            fprintf('Wrong angle MSE percentage: %.2f%%\n', fold_wrong_angle_MSE_prcnt);
        end
        fprintf('Fold time: %.2f seconds\n', fold_train_time(fold));
        
        % Save bad trajectories only for first fold
        if fold == 1 && not(validation)
            % Create directory if it doesn't exist
            if ~exist('test_data', 'dir')
                mkdir('test_data');
            end
            if ~exist('test_data/bad_trajectory', 'dir')
                mkdir('test_data/bad_trajectory');
            end
            if ~exist('test_data/bad_angle', 'dir')
                mkdir('test_data/bad_angle');
            end
            
            % Save bad trajectories to txt files
            for i = 1:length(bad_trajectories)
                filename = sprintf('test_data/bad_trajectory/%s_seed%d_bad_traj_trial%d_dir%d.txt', model, seed, bad_trajectories(i).trial, bad_trajectories(i).direction);
                fid = fopen(filename, 'w');
                fprintf(fid, 'Trial: %d, Direction: %d, MSE: %.2f\n', bad_trajectories(i).trial, bad_trajectories(i).direction, bad_trajectories(i).MSE);
                fprintf(fid, 'DecodedX,DecodedY,ActualX,ActualY\n');
                for j = 1:length(bad_trajectories(i).decodedX)
                    if j <= length(bad_trajectories(i).actualX)
                        fprintf(fid, '%.4f,%.4f,%.4f,%.4f\n', bad_trajectories(i).decodedX(j), bad_trajectories(i).decodedY(j), ...
                            bad_trajectories(i).actualX(j), bad_trajectories(i).actualY(j));
                    else
                        fprintf(fid, '%.4f,%.4f,,\n', bad_trajectories(i).decodedX(j), bad_trajectories(i).decodedY(j));
                    end
                end
                fclose(fid);
            end
            
            % Save bad angle trajectories to txt files
            for i = 1:length(bad_angle_trajectories)
                filename = sprintf('test_data/bad_angle/%s_seed%d_bad_angle_traj_trial%d_dir%d.txt', model, seed, bad_angle_trajectories(i).trial, bad_angle_trajectories(i).direction);
                fid = fopen(filename, 'w');
                fprintf(fid, 'Trial: %d, Direction: %d, MSE: %.2f\n', bad_angle_trajectories(i).trial, bad_angle_trajectories(i).direction, bad_angle_trajectories(i).MSE);
                fprintf(fid, 'DecodedX,DecodedY,ActualX,ActualY\n');
                for j = 1:length(bad_angle_trajectories(i).decodedX)
                    if j <= length(bad_angle_trajectories(i).actualX)
                        fprintf(fid, '%.4f,%.4f,%.4f,%.4f\n', bad_angle_trajectories(i).decodedX(j), bad_angle_trajectories(i).decodedY(j), ...
                            bad_angle_trajectories(i).actualX(j), bad_angle_trajectories(i).actualY(j));
                    else
                        fprintf(fid, '%.4f,%.4f,,\n', bad_angle_trajectories(i).decodedX(j), bad_angle_trajectories(i).decodedY(j));
                    end
                end
                fclose(fid);
            end
            
            if visualize
                plotSavedTrajectories(bad_trajectories, 'Bad trajectories', 0.2);
                plotSavedTrajectories(bad_angle_trajectories, 'Bad angle trajectories', 0.2);
            end
        end
    end

    %% Calculate average metrics across all folds
    train_time = mean(fold_train_time);
    val_time = mean(fold_val_time);
    time = train_time + val_time;
    RMSE = mean(fold_RMSE);
    MSE = mean(fold_MSE);
    n_predictions = sum(fold_n_predictions);
    
    if ismember(model, angle_models)
        correct_angle_predictions = sum(fold_correct_angle_predictions);
        angle_predictions = sum(fold_angle_predictions);
        wrong_angle_MSE = mean(fold_wrong_angle_MSE);
        wrong_angle_indices = sum(fold_wrong_angle_indices, 2);
    else
        correct_angle_predictions = 0;
        angle_predictions = 0;
        wrong_angle_MSE = 0;
        wrong_angle_indices = zeros(8, 1);
    end
    
    % Calculate final MSE per timestep and correct angle per timestep
    MSE_per_timestep = agg_MSE_per_timestep;
    correct_angle_per_timestep = agg_correct_angle_per_timestep;
    
    %% Report average results across all folds
    fprintf('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nCross-validation Complete\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');
    fprintf('Average results across %d folds:\n', folds);
    fprintf('Average RMSE: %.2f (std: %.2f)\n', RMSE, std(fold_RMSE));
    fprintf('Average MSE: %.2f (std: %.2f)\n', MSE, std(fold_MSE));
    fprintf('Number of predictions: %.2f\n', n_predictions);
    
    if ismember(model, angle_models)
        correct_angle_prcnt = correct_angle_predictions/angle_predictions*100;
        wrong_angle_MSE_prcnt = (wrong_angle_MSE/MSE*100);
        fprintf('Average correct angle predictions: %.2f (std: %.2f)\n', correct_angle_predictions, std(fold_correct_angle_predictions));
        fprintf('Average angle predictions: %.2f (std: %.2f)\n', angle_predictions, std(fold_angle_predictions));
        fprintf('Average correct angle predictions percentage: %.2f%%\n', correct_angle_prcnt);
        fprintf('Average wrong angle MSE: %.2f (std: %.2f)\n', wrong_angle_MSE, std(fold_wrong_angle_MSE));
        fprintf('Average wrong angle MSE percentage: %.2f%%\n', wrong_angle_MSE_prcnt);
    end
    
    fprintf('Average time per fold: %.2f seconds\n', mean(fold_train_time));
    fprintf('Total time: %.2f seconds\n', time);
    
    %% Plot average MSE per timestep
    if not(validation)
        avg_MSE_per_timestep = zeros(timesteps, 1);
        for t = 1:timesteps
            if ~isempty(MSE_per_timestep{t})
                avg_MSE_per_timestep(t) = mean(MSE_per_timestep{t});
            else
                avg_MSE_per_timestep(t) = NaN;
            end
        end
        
        figure('Position', [400 50 1200 950])
        bar(first_timestep:20:(first_timestep + (timesteps-1)*20), avg_MSE_per_timestep)
        grid on
        xticks(first_timestep:20:(first_timestep + (timesteps-1)*20))
        xlabel('Timestep')
        ylabel('Average MSE')
        title(sprintf('Average MSE per timestep across %d folds', folds))
    end

    fprintf('\n\n\n')
end

%% Helper functions
function toggle_pause(src, ~)
    global is_paused;
    is_paused = get(src, 'Value');
    
    if is_paused
        set(src, 'String', 'Resume');
    else
        set(src, 'String', 'Pause');
    end
end

function check_pause()
    global is_paused;
    while is_paused
        pause(0.1);
        drawnow;
    end
end

function plotSavedTrajectories(trajectories, title_text, pause_len)
    if isempty(trajectories)
        return;
    end
    
    numTrajectories = length(trajectories);
    cols = ceil(sqrt(numTrajectories));
    rows = ceil(numTrajectories / cols);
    
    mainFig = figure('Position', [50, 50, 1800, 900]);
    
    subplots = zeros(numTrajectories, 1);
    decodedLines = zeros(numTrajectories, 1);
    actualLines = zeros(numTrajectories, 1);
    
    for i = 1:numTrajectories
        subplots(i) = subplot(rows, cols, i);
        hold on;
        decodedLines(i) = plot(trajectories(i).decodedX, trajectories(i).decodedY, 'r-', 'LineWidth', 1.5);
        actualLines(i) = plot(trajectories(i).actualX, trajectories(i).actualY, 'b-', 'LineWidth', 1.5);
        set(decodedLines(i), 'XData', [], 'YData', []);
        set(actualLines(i), 'XData', [], 'YData', []);
        grid on;
        title(sprintf('Trial: %d, Direction: %d, MSE: %.2f', trajectories(i).trial, trajectories(i).direction, trajectories(i).MSE));
        xlabel('X Position');
        ylabel('Y Position');
        axis square;
        
        allX = [trajectories(i).decodedX(:); trajectories(i).actualX(:)];
        allY = [trajectories(i).decodedY(:); trajectories(i).actualY(:)];
        xlim([min(allX), max(allX)]);
        ylim([min(allY), max(allY)]);
        
        hold off;
    end
    sgtitle([title_text ' (Decoded - red, actual - blue)']);
    
    animButton = uicontrol('Parent', mainFig, 'Style', 'pushbutton', 'String', 'Animate Trajectories', ...
                           'Position', [50, 20, 150, 30], 'Callback', @animateTrajectories);
    
    function animateTrajectories(~, ~)
        animButton.Enable = 'off';
        
        maxLength = 0;
        for j = 1:numTrajectories
            maxLength = max(maxLength, length(trajectories(j).decodedX));
        end
        
        % Animation loop
        for step = 1:maxLength
            for j = 1:numTrajectories
                lenDecoded = length(trajectories(j).decodedX);
                lenActual = length(trajectories(j).actualY);
                
                if step <= lenDecoded
                    set(decodedLines(j), 'XData', trajectories(j).decodedX(1:step), ...
                                        'YData', trajectories(j).decodedY(1:step));
                end
                
                if step <= lenActual
                    set(actualLines(j), 'XData', trajectories(j).actualX(1:step), ...
                                       'YData', trajectories(j).actualY(1:step));
                end
            end
            
            drawnow;
            pause(pause_len);
        end
        
        animButton.Enable = 'on';
    end
end