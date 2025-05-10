%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Team: BeeAmI
% Authors:
% Niel De Backer, Felix Verstraete, Cova Coll Brugarolas, Szymon Modrzynski
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x, y, modelParameters, angle] = positionEstimator(test_data, modelParameters, arg)
    
    persistent classified_angle last_trial_id last_pos velocity_history;
    
    prob_multiplier = 1;
    alpha = 0.55; % by validation
    angle_confidence_threshold = 0.7;
    bin_size = modelParameters.bin_size;
    prep_window = modelParameters.prep_window;
    window_size = modelParameters.window_size;
    current_length = size(test_data.spikes, 2);

    if ~isfield(modelParameters, 'classified_angles')
        modelParameters.classified_angles = zeros(8, 1);
    end
    
    if ~isfield(modelParameters, 'trial_angles')
        modelParameters.trial_angles = cell(1, 100);
    end
    
    % Reset persistent variables for new trial
    if isempty(classified_angle) || test_data.trialId ~= last_trial_id
        classified_angle = 0;
        last_trial_id = test_data.trialId;
        last_pos = test_data.startHandPos;
        velocity_history = zeros(2, 20);
    end

    % Angle classification
    if classified_angle == 0 && current_length >= prep_window
        % Calculate firing rates
        fr = sum(test_data.spikes(:, 1:prep_window), 2) / prep_window * 1000;
        
        % Apply PCA transform
        pca_components = modelParameters.pca_components;
        pca_features = pca_components' * fr;
        
        % Get starting position
        startPos = test_data.startHandPos;
        
        % Combine features: PCA features, firing rates, and starting position
        combined_features = [pca_features; fr; startPos];
        
        % LDA classification
        scores = modelParameters.lda_weights * combined_features + modelParameters.lda_bias;
        
        % Calculate score adjustments
        classified_angles = modelParameters.classified_angles;
        total_counts = sum(classified_angles) + 8;
        inv_probs = (total_counts - classified_angles) / total_counts;
        norm_multipliers = prob_multiplier * (inv_probs / max(inv_probs));
        adjusted_scores = scores .* norm_multipliers;

        % Find unique angle
        [sorted_scores, sorted_indices] = sort(adjusted_scores, 'descend');
        classified_angle = 0;
        
        % Get current trial data
        trial_id = test_data.trialId;
        trial_angles = modelParameters.trial_angles;
        
        % Check confidence margin between top two scores
        if length(sorted_scores) >= 2 && (sorted_scores(1) - sorted_scores(2)) < modelParameters.confidence_margin_threshold
            % Close margin -> check for problem angles
            top_angles = sorted_indices(1:min(3, length(sorted_indices)));
            problem_indices = find(ismember(top_angles, modelParameters.problem_angles));
            
            if ~isempty(problem_indices)
                % Use specialized classifiers for problem angles
                specialized_scores = zeros(length(problem_indices), 1);
                
                for i = 1:length(problem_indices)
                    pa_idx = find(modelParameters.problem_angles == top_angles(problem_indices(i)));
                    if ~isempty(pa_idx)
                        classifier = modelParameters.specialized_classifiers{pa_idx};
                        specialized_scores(i) = classifier.weights' * combined_features + classifier.bias;
                    end
                end
                
                % If strong response
                if any(specialized_scores > angle_confidence_threshold)
                    [~, max_idx] = max(specialized_scores);
                    classified_angle = top_angles(problem_indices(max_idx));
                    
                    % Check if this angle is already used
                    if trial_id <= length(trial_angles) && ~isempty(trial_angles{trial_id}) && any(trial_angles{trial_id} == classified_angle)
                        % Use pairwise classifier as fallback
                        top_angle1 = sorted_indices(1);
                        top_angle2 = sorted_indices(2);
                        
                        if top_angle1 > top_angle2
                            temp = top_angle1;
                            top_angle1 = top_angle2;
                            top_angle2 = temp;
                        end
                        
                        if top_angle1 < top_angle2 && ~isempty(modelParameters.pairwise_classifiers{top_angle1, top_angle2})
                            pair_classifier = modelParameters.pairwise_classifiers{top_angle1, top_angle2};
                            pair_scores = pair_classifier.weights * combined_features + pair_classifier.bias;
                            [~, pair_winner_idx] = max(pair_scores);
                            pair_winner = pair_classifier.angles(pair_winner_idx);
                            
                            if ~any(trial_angles{trial_id} == pair_winner)
                                classified_angle = pair_winner;
                            end
                        end
                    end
                else
                    % Use confusion weights to adjust scores
                    confusion_adj = zeros(size(top_angles));
                    
                    for i = 1:length(top_angles)
                        a = top_angles(i);
                        confusion_sum = 0;
                        for j = 1:length(top_angles)
                            if i ~= j
                                a_other = top_angles(j);
                                confusion_sum = confusion_sum + modelParameters.confusion_weights(a_other, a);
                            end
                        end
                        confusion_adj(i) = -confusion_sum; % penalize angles that are often confused
                    end
                    
                    adjusted_top_scores = sorted_scores(1:length(top_angles)) + confusion_adj * 0.2;
                    [~, best_idx] = max(adjusted_top_scores);
                    classified_angle = top_angles(best_idx);
                    
                    % Check if this angle is already used
                    if trial_id <= length(trial_angles) && ~isempty(trial_angles{trial_id}) && any(trial_angles{trial_id} == classified_angle)
                        % Go next
                        adjusted_top_scores(best_idx) = -Inf;
                        [~, next_best_idx] = max(adjusted_top_scores);
                        if next_best_idx <= length(top_angles)
                            classified_angle = top_angles(next_best_idx);
                        end
                    end
                end
            else
                % Pairwise classification for non-problem angles
                top_angle1 = sorted_indices(1);
                top_angle2 = sorted_indices(2);

                % Use pairwise classifier
                pair_classifier = modelParameters.pairwise_classifiers{top_angle1, top_angle2};
                pair_scores = pair_classifier.weights * combined_features + pair_classifier.bias;
                [~, pair_winner_idx] = max(pair_scores);
                
                % Get the actual angle from the pairwise angles
                pair_winner = pair_classifier.angles(pair_winner_idx);
                
                % Check if we need to search for unique angle
                if trial_id > length(trial_angles) || isempty(trial_angles{trial_id}) || ~any(trial_angles{trial_id} == pair_winner)
                    classified_angle = pair_winner;
                else
                    % Try other angle
                    other_angle_idx = 3 - pair_winner_idx;
                    other_angle = pair_classifier.angles(other_angle_idx);
                    
                    if ~any(trial_angles{trial_id} == other_angle)
                        classified_angle = other_angle;
                    else
                        % Original sorting logic as fallback
                        for i = 1:length(sorted_scores)
                            candidate = sorted_indices(i);
                            if ~any(trial_angles{trial_id} == candidate)
                                classified_angle = candidate;
                                break;
                            end
                        end
                    end
                end
            end
        else
            % High confidence -> original method
            for i = 1:length(sorted_scores)
                candidate = sorted_indices(i);
                if trial_id > length(trial_angles) || isempty(trial_angles{trial_id}) || ~any(trial_angles{trial_id} == candidate)
                    classified_angle = candidate;
                    break;
                end
            end
        end

        % Default to highest score if all angles classified somehow
        if classified_angle == 0
            [~, classified_angle] = max(adjusted_scores);
        end
        
        angle = classified_angle;
        
        % Update model state
        modelParameters.classified_angles(classified_angle) = modelParameters.classified_angles(classified_angle) + 1;
        
        % Update trial angles
        if trial_id > length(trial_angles)
            modelParameters.trial_angles{trial_id} = classified_angle;
        else
            modelParameters.trial_angles{trial_id} = [modelParameters.trial_angles{trial_id}, classified_angle];
        end
    else
        angle = classified_angle;
    end
    
    % Trajectory-based prediction
    movement_length = current_length - prep_window;
    current_bin = floor(movement_length / bin_size) + 1;
    max_bins_value = max(modelParameters.max_bins);
    
    if current_bin > max_bins_value
        current_bin = max_bins_value;
    end
    
    % Initialize with start position
    startPos = test_data.startHandPos;
    traj_x = startPos(1);
    traj_y = startPos(2);
    
    % Trajectory prediction
    if classified_angle > 0 && classified_angle <= length(modelParameters.avg_trajectories)
        avg_traj = modelParameters.avg_trajectories{classified_angle};
        if current_bin <= size(avg_traj, 2)
            predicted_rel_pos = avg_traj(:, current_bin);
        else
            predicted_rel_pos = avg_traj(:, end);
        end
        
        traj_x = startPos(1) + predicted_rel_pos(1);
        traj_y = startPos(2) + predicted_rel_pos(2);
    end
    
    % Velocity-based prediction
    % Use pre-calculated indices for recent spikes
    recent_start = current_length - window_size + 1;
    recent_spikes = test_data.spikes(:, recent_start:current_length);
    
    % Calculate firing rate
    fr = sum(recent_spikes, 2) * (1000/window_size);
    
    % Predict velocity
    X_vel_centered = fr - modelParameters.X_vel_mean;
    predicted_vel = modelParameters.vel_beta * X_vel_centered + modelParameters.Y_vel_mean;

    % Update velocity history
    velocity_history(:, 1:end-1) = velocity_history(:, 2:end);
    velocity_history(:, end) = predicted_vel;
    
    smoothed_vel = sum(velocity_history, 2) / size(velocity_history, 2);
    
    % Update position based on velocity
    vel_x = last_pos(1) + smoothed_vel(1);
    vel_y = last_pos(2) + smoothed_vel(2);
    
    % Still use the trajectory positions as that's what validation confirmed
    last_pos = [traj_x; traj_y];

    % Adaptive weighting of predictions
    adapt_alpha = alpha;
    if current_bin > 10
        adapt_alpha = max(0.3, alpha - 0.02 * (current_bin - 10));
    end
    
    % Calculate final position
    one_minus_alpha = 1 - adapt_alpha;
    x = adapt_alpha * traj_x + one_minus_alpha * vel_x;
    y = adapt_alpha * traj_y + one_minus_alpha * vel_y;
    
    % Handle invalid values
    if isnan(x) || isnan(y) || isinf(x) || isinf(y)
        x = startPos(1);
        y = startPos(2);
        last_pos = startPos;
    end
end
