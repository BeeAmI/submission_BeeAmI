function validationScript(models, runs, basic_visualisation, folds)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Parts of the code generated with Github Copilot.
    % Run test script runs times with seed corresponding
    % to the run number + number of cross-validation folds.
    % Print out & plot some stats.
    % args:
    % models: ({string}) folder name of the models under /BeeAmI
    % runs: (int) number of times testScript is run
    % basic_visualisation: (boolean) whether to show the main visualisation plot
    % folds: (int) number of folds
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    clc;
    close all;
    warning('off', 'all');

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
    colors = {'#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F'};

    results = cell(length(models), 1);

    first_timestep = 320;
    
    for m = 1:length(models)
        model = models{m};

        times = zeros(runs, 1);
        train_times = zeros(runs, 1);
        val_times = zeros(runs, 1);
        timesteps = 0;
        RMSEs = zeros(runs, 1);
        MSEs = zeros(runs, 1);
        all_n_predictions = zeros(runs, 1);
        wrong_angle_MSEs = zeros(runs, 1);
        total_wrong_angle_indices = zeros(8, 1);
        avg_MSEs_per_timestep = cell(runs, 1);
        MSEs_per_timestep = cell(runs, 1);
        correct_angle_per_timestep = cell(runs, 1);
        all_correct_angle_predictions = zeros(runs, 1);
        all_angle_predictions = zeros(runs, 1);

        for r = 1:runs
            [RMSE, MSE, n_predictions, MSE_per_timestep, out_correct_angle_per_timestep,...
             correct_angle_predictions, angle_predictions, wrong_angle_indices, ...
              wrong_angle_MSE, time, train_time, val_time] = testScript(model, basic_visualisation, false, r, true, folds);
            RMSEs(r) = RMSE;
            MSEs(r) = MSE;
            times(r) = time;
            train_times(r) = train_time;
            val_times(r) = val_time;
            all_n_predictions(r) = n_predictions;
            wrong_angle_MSEs(r) = wrong_angle_MSE;
            total_wrong_angle_indices = total_wrong_angle_indices + wrong_angle_indices;
            all_correct_angle_predictions(r) = correct_angle_predictions;
            all_angle_predictions(r) = angle_predictions;
            avg_MSE_per_timestep = zeros(length(MSE_per_timestep),1);
            for t = 1:length(MSE_per_timestep)
                current_values = MSE_per_timestep{t};
                current_values(isnan(current_values)) = 0;
                avg_MSE_per_timestep(t) = mean(current_values);
            end
            avg_MSEs_per_timestep{r} = avg_MSE_per_timestep;
            MSEs_per_timestep{r} = MSE_per_timestep;
            correct_angle_per_timestep{r} = out_correct_angle_per_timestep;
            if length(avg_MSE_per_timestep) > timesteps
                timesteps = length(avg_MSE_per_timestep);
            end
        end

        %% Variable Preparation
        combined_avg_MSEs = zeros(timesteps, 1);
        combined_avg_MSEs_correct_angles = zeros(timesteps, 1);
        combined_avg_MSEs_wrong_angles = zeros(timesteps, 1);
        
        all_avg_MSEs = zeros(timesteps, 1);
        all_avg_RMSEs = zeros(timesteps, 1);
        idx = 1;
        
        % Count how many samples per timestep and angle category
        correct_angle_count = zeros(timesteps, 1);
        wrong_angle_count = zeros(timesteps, 1);
        
        for r = 1:runs
            for t = 1:length(MSEs_per_timestep{r})
            current_value = avg_MSEs_per_timestep{r}(t);
            if ~isnan(current_value) && current_value ~= 0
                all_avg_MSEs(idx) = current_value;
                all_avg_RMSEs(idx) = sqrt(current_value);
            end
            idx = idx + 1;
            
            if ismember(model, angle_models)
                for i = 1:length(MSEs_per_timestep{r}{t})
                if correct_angle_per_timestep{r}{t}(i) == true
                    combined_avg_MSEs_correct_angles(t) = combined_avg_MSEs_correct_angles(t) + MSEs_per_timestep{r}{t}(i);
                    correct_angle_count(t) = correct_angle_count(t) + 1;
                else
                    combined_avg_MSEs_wrong_angles(t) = combined_avg_MSEs_wrong_angles(t) + MSEs_per_timestep{r}{t}(i);
                    wrong_angle_count(t) = wrong_angle_count(t) + 1;
                end
                end
            end
            end
        end
        
        % Compute averages
        if ismember(model, angle_models)
            for t = 1:timesteps
            if correct_angle_count(t) > 0
                combined_avg_MSEs_correct_angles(t) = combined_avg_MSEs_correct_angles(t) / correct_angle_count(t);
            end
            if wrong_angle_count(t) > 0
                combined_avg_MSEs_wrong_angles(t) = combined_avg_MSEs_wrong_angles(t) / wrong_angle_count(t);
            end
            
            % Calculate combined_avg_MSEs from correct and wrong angles
            total_count = correct_angle_count(t) + wrong_angle_count(t);
            if total_count > 0
                combined_avg_MSEs(t) = (combined_avg_MSEs_correct_angles(t) * correct_angle_count(t) + ...
                          combined_avg_MSEs_wrong_angles(t) * wrong_angle_count(t)) / total_count;
            end
            end
        else
            % For non-angle models, calculate combined_avg_MSEs directly
            for r = 1:runs
            for t = 1:length(MSEs_per_timestep{r})
                current_value = avg_MSEs_per_timestep{r}(t);
                if ~isnan(current_value) && current_value ~= 0
                combined_avg_MSEs(t) = combined_avg_MSEs(t) + current_value/runs;
                end
            end
            end
        end

        avg_time = mean(times);
        avg_train_time = mean(train_times);
        avg_val_time = mean(val_time);
        avg_RMSE = mean(RMSEs);
        std_RMSE = std(RMSEs);
        min_RMSE = min(RMSEs);
        max_RMSE = max(RMSEs);
        avg_RMSE_timestep = mean(all_avg_RMSEs);
        std_RMSE_timestep = std(all_avg_RMSEs);
        avg_MSE_timestep = mean(all_avg_MSEs);
        std_MSE_timestep = std(all_avg_MSEs);
        avg_total_MSE = mean(MSEs);
        sum_total_MSEs = sum(MSEs);
        sum_all_n_predictions = sum(all_n_predictions);
        avg_wrong_angle_MSEs = mean(wrong_angle_MSEs);
        sum_wrong_angle_MSEs = sum(wrong_angle_MSEs);
        wrong_angle_MSE_percentage = sum_wrong_angle_MSEs/sum_total_MSEs*100;
        sum_all_correct_angle_predictions = sum(all_correct_angle_predictions);
        sum_all_angle_predictions = sum(all_angle_predictions);
        all_correct_angle_predictions_percentage = sum_all_correct_angle_predictions/sum_all_angle_predictions*100;
        % simulate the airjet
        % time_score_multiplier = 1.5;
        time_score = avg_time * 0.1;
        RMSE_score = avg_RMSE * 0.9;
        competition_score = RMSE_score + time_score;

        results{m} = struct(...
            'model', model, ...
            'RMSEs', RMSEs, ...
            'MSEs', MSEs, ...
            'train_times', train_times, ...
            'val_times', val_times, ...
            'times', times, ...
            'all_n_predictions', all_n_predictions, ...
            'wrong_angle_MSEs', wrong_angle_MSEs, ...
            'total_wrong_angle_indices', total_wrong_angle_indices, ...
            'all_correct_angle_predictions', all_correct_angle_predictions, ...
            'all_angle_predictions', all_angle_predictions, ...
            'avg_MSEs_per_timestep', {avg_MSEs_per_timestep}, ...
            'MSEs_per_timestep', {MSEs_per_timestep}, ...
            'correct_angle_per_timestep', {correct_angle_per_timestep}, ...
            'combined_avg_MSEs', combined_avg_MSEs, ...
            'combined_avg_MSEs_correct_angles', combined_avg_MSEs_correct_angles, ...
            'combined_avg_MSEs_wrong_angles', combined_avg_MSEs_wrong_angles, ...
            'avg_train_time', avg_train_time, ...
            'avg_val_time', avg_val_time, ...
            'avg_time', avg_time, ...
            'avg_RMSE', avg_RMSE, ...
            'std_RMSE', std_RMSE, ...
            'min_RMSE', min_RMSE, ...
            'max_RMSE', max_RMSE, ...
            'avg_RMSE_timestep', avg_RMSE_timestep, ...
            'std_RMSE_timestep', std_RMSE_timestep, ...
            'avg_MSE_timestep', avg_MSE_timestep, ...
            'std_MSE_timestep', std_MSE_timestep, ...
            'avg_total_MSE', avg_total_MSE, ...
            'time_score', time_score, ...
            'RMSE_score', RMSE_score, ...
            'competition_score', competition_score, ...
            'is_angle_model', ismember(model, angle_models), ...
            'sum_all_n_predictions', sum_all_n_predictions, ...
            'sum_wrong_angle_MSEs', sum_wrong_angle_MSEs, ...
            'wrong_angle_MSE_percentage', wrong_angle_MSE_percentage, ...
            'sum_all_correct_angle_predictions', sum_all_correct_angle_predictions, ...
            'sum_all_angle_predictions', sum_all_angle_predictions, ...
            'all_correct_angle_predictions_percentage', all_correct_angle_predictions_percentage);

        %% Plotting
        ceiling = max([max(combined_avg_MSEs), max(combined_avg_MSEs_correct_angles)]) + 100;
        color = colors{mod(m-1, length(colors)) + 1};

        figure('Position', [400 50 1200 950])

        subplot(3,1,1)
        bar(first_timestep:20:(300 + timesteps*20), combined_avg_MSEs, 'FaceColor', color)
        grid on
        xticks(first_timestep:20:(300 + timesteps*20))
        xlabel('Timestep')
        ylim([0, ceiling])
        % yticks(0:100:ceiling+1)
        ylabel('Average MSE')
        title(['Average MSE per timestep'])

        if ismember(model, angle_models)
            subplot(3,1,2)
            bar(first_timestep:20:(300 + timesteps*20), combined_avg_MSEs_correct_angles, 'FaceColor', color)
            grid on
            xticks(first_timestep:20:(300 + timesteps*20))
            xlabel('Timestep')
            ylim([0, ceiling])
            % yticks(0:100:ceiling+1)
            ylabel('Average MSE')
            title(['Average correct angle MSE per timestep'])

            subplot(3,1,3)
            bar(first_timestep:20:(300 + timesteps*20), combined_avg_MSEs_wrong_angles, 'FaceColor', color)
            grid on
            xticks(first_timestep:20:(300 + timesteps*20))
            xlabel('Timestep')
            ylim([0, max(combined_avg_MSEs_wrong_angles) + 100])
            % yticks(0:100:ceiling+1)
            ylabel('Average MSE')
            title(['Average wrong angle MSE per timestep'])
        end

        sgtitle(['MSE for ' strrep(model, '_', '-'), ' model'], 'FontSize', 14, 'FontWeight', 'bold')
    end

    %% Printing Results as Table
    fprintf('\n\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\nValidation Results\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
    
    % Define metrics to display
    metrics = {'Number of all predictions', 'Training time (seconds)', 'Validation time (seconds)', 'Average time (seconds)', 'Average RMSE', 'Std RMSE', ...
               'Min average RMSE', 'Max average RMSE', 'Average MSE per timestep', 'Std MSE per timestep', 'Average total MSE'}; % 'Average RMSE per timestep', 'Std RMSE per timestep', 
    
    % Check if any model is an angle model to include angle-specific metrics
    any_angle_model = false;
    for m = 1:length(models)
        if results{m}.is_angle_model
            any_angle_model = true;
            break;
        end
    end
    
    if any_angle_model
        metrics = [metrics, {'Avg wrong angle total MSE', 'Wrong angle MSE percentage (%)', ...
                  'Total correct angle predictions', 'Total angle predictions', ...
                  'Correct angle percentage (%)', 'Wrong angle indices', '1.', '2.', ...
                  '3.', '4.', '5.', '6.', '7.', '8.'}];
    end
    
    metrics = [metrics, {'Competition time score', 'Competition RMSE score', 'Competition score'}];
    
    % Determine the width for each column
    metric_width = 35;
    value_width = 15;
    
    % Print header
    fprintf('%-*s', metric_width, 'Metric');
    for m = 1:length(models)
        fprintf('| %-*s', value_width, results{m}.model);
    end
    fprintf('\n');
    
    % Print separator line
    separator = repmat('-', 1, metric_width);
    fprintf('%-*s', metric_width, separator);
    for m = 1:length(models)
        fprintf('|%s', repmat('-', 1, value_width+1));
    end
    fprintf('\n');
    
    % Print each metric
    for i = 1:length(metrics)
        fprintf('%-*s', metric_width, metrics{i});
        
        for m = 1:length(models)
            switch metrics{i}
                case 'Number of all predictions'
                    value = results{m}.sum_all_n_predictions;
                    format = '| %-*d';
                case 'Training time (seconds)'
                    value = results{m}.avg_train_time;
                    format = '| %-*.2f';
                case 'Validation time (seconds)'
                    value = results{m}.avg_val_time;
                    format = '| %-*.2f';
                case 'Average time (seconds)'
                    value = results{m}.avg_time;
                    format = '| %-*.2f';
                case 'Average RMSE'
                    value = results{m}.avg_RMSE;
                    format = '| %-*.2f';
                case 'Std RMSE'
                    value = results{m}.std_RMSE;
                    format = '| %-*.2f';
                case 'Min average RMSE'
                    value = results{m}.min_RMSE;
                    format = '| %-*.2f';
                case 'Max average RMSE'
                    value = results{m}.max_RMSE;
                    format = '| %-*.2f';
                % case 'Average RMSE per timestep'
                %     value = results{m}.avg_RMSE_timestep;
                %     format = '| %-*.2f';
                % case 'Std RMSE per timestep'
                %     value = results{m}.std_RMSE_timestep;
                %     format = '| %-*.2f';
                case 'Average MSE per timestep'
                    value = results{m}.avg_MSE_timestep;
                    format = '| %-*.2f';
                case 'Std MSE per timestep'
                    value = results{m}.std_MSE_timestep;
                    format = '| %-*.2f';
                case 'Average total MSE'
                    value = results{m}.avg_total_MSE;
                    format = '| %-*.2f';
                case 'Avg wrong angle total MSE'
                    if results{m}.is_angle_model
                        value = mean(results{m}.wrong_angle_MSEs);
                    else
                        value = 'N/A';
                        format = '| %-*s';
                    end
                    format = '| %-*.2f';
                case 'Wrong angle MSE percentage (%)'
                    if results{m}.is_angle_model
                        value = results{m}.wrong_angle_MSE_percentage;
                    else
                        value = 'N/A';
                        format = '| %-*s';
                    end
                    format = '| %-*.2f';
                case 'Total correct angle predictions'
                    if results{m}.is_angle_model
                        value = results{m}.sum_all_correct_angle_predictions;
                    else
                        value = 'N/A';
                        format = '| %-*s';
                    end
                    format = '| %-*d';
                case 'Total angle predictions'
                    if results{m}.is_angle_model
                        value = results{m}.sum_all_angle_predictions;
                    else
                        value = 'N/A';
                        format = '| %-*s';
                    end
                    format = '| %-*d';
                case 'Correct angle percentage (%)'
                    if results{m}.is_angle_model
                        value = results{m}.all_correct_angle_predictions_percentage;
                    else
                        value = 'N/A';
                        format = '| %-*s';
                    end
                    format = '| %-*.2f';
                case 'Wrong angle indices'
                    if results{m}.is_angle_model
                        value = '';
                    else
                        value = 'N/A';
                        format = '| %-*s';
                    end
                    format = '| %-*d';
                case '1.'
                    if results{m}.is_angle_model
                        value = results{m}.total_wrong_angle_indices(1);
                        percentage = (results{m}.total_wrong_angle_indices(1) / (results{m}.sum_all_angle_predictions - results{m}.sum_all_correct_angle_predictions)) * 100;
                        display_value = sprintf('%g (%.1f%%)', value, percentage);
                        format = '| %-*s';
                    else
                        display_value = 'N/A';
                        format = '| %-*s';
                    end
                    fprintf(format, value_width, display_value);
                    continue;
                case '2.'
                    if results{m}.is_angle_model
                        value = results{m}.total_wrong_angle_indices(2);
                        percentage = (results{m}.total_wrong_angle_indices(2) / (results{m}.sum_all_angle_predictions - results{m}.sum_all_correct_angle_predictions)) * 100;
                        display_value = sprintf('%g (%.1f%%)', value, percentage);
                        format = '| %-*s';
                    else
                        display_value = 'N/A';
                        format = '| %-*s';
                    end
                    fprintf(format, value_width, display_value);
                    continue;
                case '3.'
                    if results{m}.is_angle_model
                        value = results{m}.total_wrong_angle_indices(3);
                        percentage = (results{m}.total_wrong_angle_indices(3) / (results{m}.sum_all_angle_predictions - results{m}.sum_all_correct_angle_predictions)) * 100;
                        display_value = sprintf('%g (%.1f%%)', value, percentage);
                        format = '| %-*s';
                    else
                        display_value = 'N/A';
                        format = '| %-*s';
                    end
                    fprintf(format, value_width, display_value);
                    continue;
                case '4.'
                    if results{m}.is_angle_model
                        value = results{m}.total_wrong_angle_indices(4);
                        percentage = (results{m}.total_wrong_angle_indices(4) / (results{m}.sum_all_angle_predictions - results{m}.sum_all_correct_angle_predictions)) * 100;
                        display_value = sprintf('%g (%.1f%%)', value, percentage);
                        format = '| %-*s';
                    else
                        display_value = 'N/A';
                        format = '| %-*s';
                    end
                    fprintf(format, value_width, display_value);
                    continue;
                case '5.'
                    if results{m}.is_angle_model
                        value = results{m}.total_wrong_angle_indices(5);
                        percentage = (results{m}.total_wrong_angle_indices(5) / (results{m}.sum_all_angle_predictions - results{m}.sum_all_correct_angle_predictions)) * 100;
                        display_value = sprintf('%g (%.1f%%)', value, percentage);
                        format = '| %-*s';
                    else
                        display_value = 'N/A';
                        format = '| %-*s';
                    end
                    fprintf(format, value_width, display_value);
                    continue;
                case '6.'
                    if results{m}.is_angle_model
                        value = results{m}.total_wrong_angle_indices(6);
                        percentage = (results{m}.total_wrong_angle_indices(6) / (results{m}.sum_all_angle_predictions - results{m}.sum_all_correct_angle_predictions)) * 100;
                        display_value = sprintf('%g (%.1f%%)', value, percentage);
                        format = '| %-*s';
                    else
                        display_value = 'N/A';
                        format = '| %-*s';
                    end
                    fprintf(format, value_width, display_value);
                    continue;
                case '7.'
                    if results{m}.is_angle_model
                        value = results{m}.total_wrong_angle_indices(7);
                        percentage = (results{m}.total_wrong_angle_indices(7) / (results{m}.sum_all_angle_predictions - results{m}.sum_all_correct_angle_predictions)) * 100;
                        display_value = sprintf('%g (%.1f%%)', value, percentage);
                        format = '| %-*s';
                    else
                        display_value = 'N/A';
                        format = '| %-*s';
                    end
                    fprintf(format, value_width, display_value);
                    continue;
                case '8.'
                    if results{m}.is_angle_model
                        value = results{m}.total_wrong_angle_indices(8);
                        percentage = (results{m}.total_wrong_angle_indices(8) / (results{m}.sum_all_angle_predictions - results{m}.sum_all_correct_angle_predictions)) * 100;
                        display_value = sprintf('%g (%.1f%%)', value, percentage);
                        format = '| %-*s';
                    else
                        display_value = 'N/A';
                        format = '| %-*s';
                    end
                    fprintf(format, value_width, display_value);
                    continue;
                case 'Competition time score'
                    value = results{m}.time_score;
                    format = '| %-*.3f';
                case 'Competition RMSE score'
                    value = results{m}.RMSE_score;
                    format = '| %-*.3f';
                case 'Competition score'
                    value = results{m}.competition_score;
                    format = '| %-*.3f';
            end
            
            if ischar(value)
                fprintf('| %-*s', value_width, value);
            else
                fprintf(format, value_width, value);
            end
        end
        fprintf('\n');
    end
end
