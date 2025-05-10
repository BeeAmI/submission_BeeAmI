%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Team: BeeAmI
% Authors:
% Niel De Backer, Felix Verstraete, Cova Coll Brugarolas, Szymon Modrzynski
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function modelParameters = positionEstimatorTraining(training_data)
    num_trials = size(training_data, 1);
    num_angles = size(training_data, 2);
    num_neurons = size(training_data(1,1).spikes, 1);
    
    bin_size = 20;
    prep_window = 300 + 20;
    window_size = 20;
    num_pca_components = 13;
    angle_confidence_threshold = 0.25;
    problem_angles = [3,4,5,8];
    
    max_trajectory_length = zeros(1, num_angles);
    
    % Calculate maximum trajectory length for pre-allocation
    for angle = 1:num_angles
        for trial = 1:num_trials
            spikes = training_data(trial, angle).spikes;
            if size(spikes, 2) >= prep_window + 100
                traj_len = size(spikes, 2) - prep_window;
                if traj_len > max_trajectory_length(angle)
                    max_trajectory_length(angle) = traj_len;
                end
            end
        end
    end
    
    max_bins = ceil(max_trajectory_length / bin_size);
    total_trials = num_trials * num_angles;
    
    % Estimate sizes for pre-allocation
    est_valid_trials = 0;
    est_vel_samples = 0;
    
    for angle = 1:num_angles
        for trial = 1:num_trials
            spikes = training_data(trial, angle).spikes;
            if size(spikes, 2) >= prep_window + 100
                est_valid_trials = est_valid_trials + 1;
                est_vel_samples = est_vel_samples + max(0, size(spikes, 2) - prep_window - window_size + 1);
            end
        end
    end
    
    % Safety check to ensure we have at least some trials
    if est_valid_trials == 0
        est_valid_trials = num_trials * num_angles;
    end
    
    if est_vel_samples == 0
        est_vel_samples = 1000;
    end
    
    % Pre-allocate arrays
    all_firing_rates = zeros(num_neurons, est_valid_trials);
    all_start_positions = zeros(2, est_valid_trials);
    Y_class = zeros(1, est_valid_trials);
    X_vel = zeros(num_neurons, est_vel_samples);
    Y_vel = zeros(2, est_vel_samples);
    
    % Process data in one pass
    fr_idx = 1;
    vel_idx = 1;
    
    for angle = 1:num_angles
        avg_traj = zeros(2, max_bins(angle));
        all_trajectories = zeros(2, max_bins(angle), num_trials);
        valid_trial_count = zeros(1, max_bins(angle));
        
        for trial = 1:num_trials
            spikes = training_data(trial, angle).spikes;
            
            if size(spikes, 2) < prep_window + 100
                continue;
            end
            
            % Compute firing rates for classification
            fr = sum(spikes(:, 1:prep_window), 2) / prep_window * 1000;
            
            startPos = training_data(trial, angle).handPos(1:2, 1);
            
            if fr_idx <= size(all_firing_rates, 2)
                all_firing_rates(:, fr_idx) = fr;
                all_start_positions(:, fr_idx) = startPos;
                Y_class(fr_idx) = angle;
                fr_idx = fr_idx + 1;
            end
            
            % Process trajectory data
            pos = training_data(trial, angle).handPos(1:2, prep_window+1:end);
            start_pos = training_data(trial, angle).handPos(1:2, prep_window);
            rel_pos = pos - start_pos;
            
            num_bins = min(max_bins(angle), ceil(size(rel_pos, 2) / bin_size));
            
            for b = 1:num_bins
                bin_start = (b-1) * bin_size + 1;
                bin_end = min(b * bin_size, size(rel_pos, 2));
                
                if bin_end >= bin_start
                    all_trajectories(:, b, trial) = mean(rel_pos(:, bin_start:bin_end), 2);
                    valid_trial_count(b) = valid_trial_count(b) + 1;
                end
            end
            
            % Process velocity data in one pass
            handPos = training_data(trial, angle).handPos(1:2, :);
            spike_length = size(spikes, 2);
            
            if spike_length >= prep_window + window_size
                max_t = min(spike_length-window_size, size(handPos, 2)-window_size);
                
                for t = prep_window+1:max_t
                    % Calculate firing rate for this window
                    window_fr = sum(spikes(:, t:t+window_size-1), 2) * (1000/window_size);
                    
                    % Calculate velocity as position difference
                    if t+window_size <= size(handPos, 2)
                        vel = handPos(:, t+window_size) - handPos(:, t);
                        
                        if vel_idx <= size(X_vel, 2)
                            X_vel(:, vel_idx) = window_fr;
                            Y_vel(:, vel_idx) = vel;
                            vel_idx = vel_idx + 1;
                        end
                    end
                end
            end
        end
        
        % Calculate average trajectories
        for b = 1:max_bins(angle)
            if valid_trial_count(b) > 0
                avg_traj(:, b) = sum(all_trajectories(:, b, :), 3) / valid_trial_count(b);
            end
        end
        
        avg_trajectories{angle} = avg_traj;
    end
    
    % Trim arrays to actual size
    if fr_idx > 1
        all_firing_rates = all_firing_rates(:, 1:fr_idx-1);
        all_start_positions = all_start_positions(:, 1:fr_idx-1);
        Y_class = Y_class(1:fr_idx-1);
    else
        error('No valid trials found for classification');
    end
    
    if vel_idx > 1
        X_vel = X_vel(:, 1:vel_idx-1);
        Y_vel = Y_vel(:, 1:vel_idx-1);
    else
        error('No valid velocity samples found');
    end
    
    % Apply PCA to firing rates
    all_firing_rates_centered = all_firing_rates - mean(all_firing_rates, 2);
    cov_matrix = (all_firing_rates_centered * all_firing_rates_centered') / (size(all_firing_rates, 2) - 1);
    [coeff, D] = eig(cov_matrix);
    [d, idx] = sort(diag(D), 'descend');
    coeff = coeff(:, idx);
    explained = 100 * d / sum(d);
    
    pca_components = coeff(:, 1:num_pca_components);
    
    % Compute PCA features
    pca_features = pca_components' * all_firing_rates;
    
    % Create combined features: PCA + raw firing rates
    X_class = [pca_features; all_firing_rates];
    
    % % Velocity regression
    % X_vel_mean = mean(X_vel, 2);
    % Y_vel_mean = mean(Y_vel, 2);
    
    % X_vel_centered = X_vel - X_vel_mean;
    % Y_vel_centered = Y_vel - Y_vel_mean;
    
    % % Use matrix division for better performance
    % beta = Y_vel_centered * X_vel_centered' / (X_vel_centered * X_vel_centered' + 0.01 * eye(size(X_vel, 1)));
    
    % LDA calculation
    [num_features, num_samples] = size(X_class);
    class_means = zeros(num_features, num_angles);
    class_counts = zeros(1, num_angles);
    
    % Calculate class means with vectorized operations
    for a = 1:num_angles
        idx = (Y_class == a);
        class_counts(a) = sum(idx);
        if class_counts(a) > 0
            class_means(:, a) = mean(X_class(:, idx), 2);
        end
    end
    
    % Calculate pooled covariance matrix
    S_pooled = zeros(num_features, num_features);
    for a = 1:num_angles
        idx = (Y_class == a);
        if sum(idx) <= 1
            continue;
        end
        X_centered = X_class(:, idx) - class_means(:, a);
        S_pooled = S_pooled + X_centered * X_centered';
    end
    S_pooled = S_pooled / (num_samples - num_angles);
    S_pooled = S_pooled + 0.33 * eye(num_features); % 0.33 by validation
    
    % Compute LDA weights using matrix operations
    S_inv = inv(S_pooled);
    lda_weights = zeros(num_angles, num_features);
    lda_bias = zeros(num_angles, 1);
    
    for a = 1:num_angles
        lda_weights(a, :) = (S_inv * class_means(:, a))';
        lda_bias(a) = -0.5 * class_means(:, a)' * S_inv * class_means(:, a) + log(class_counts(a)/num_samples);
    end

    % Create pairwise classifiers for each angle combination
    pairwise_classifiers = cell(num_angles, num_angles);
    
    % For each possible pair of angles
    for a1 = 1:num_angles
        for a2 = (a1+1):num_angles
            % Get data for only these two angles
            idx_pair = (Y_class == a1 | Y_class == a2);
            X_pair = X_class(:, idx_pair);
            Y_pair = Y_class(idx_pair);
            
            % Get class means for the pair
            pair_class_means = zeros(num_features, 2);
            pair_class_counts = zeros(1, 2);
            
            pair_idx1 = (Y_pair == a1);
            pair_idx2 = (Y_pair == a2);
            
            pair_class_counts(1) = sum(pair_idx1);
            pair_class_counts(2) = sum(pair_idx2);
            
            if pair_class_counts(1) > 0
                pair_class_means(:, 1) = mean(X_pair(:, pair_idx1), 2);
            end
            if pair_class_counts(2) > 0
                pair_class_means(:, 2) = mean(X_pair(:, pair_idx2), 2);
            end
            
            % Calculate covariance matrix for the pair
            pair_S_pooled = zeros(num_features, num_features);
            
            if sum(pair_idx1) > 1
                X_centered = X_pair(:, pair_idx1) - pair_class_means(:, 1);
                pair_S_pooled = pair_S_pooled + X_centered * X_centered';
            end
            
            if sum(pair_idx2) > 1
                X_centered = X_pair(:, pair_idx2) - pair_class_means(:, 2);
                pair_S_pooled = pair_S_pooled + X_centered * X_centered';
            end
            
            pair_S_pooled = pair_S_pooled / (sum(pair_idx1) + sum(pair_idx2) - 2);
            pair_S_pooled = pair_S_pooled + 0.33 * eye(num_features);  % Same regularization
            
            % Compute LDA for the pair
            pair_S_inv = inv(pair_S_pooled);
            pair_weights = zeros(2, num_features);
            pair_bias = zeros(2, 1);
            
            pair_weights(1, :) = (pair_S_inv * pair_class_means(:, 1))';
            pair_bias(1) = -0.5 * pair_class_means(:, 1)' * pair_S_inv * pair_class_means(:, 1) + log(pair_class_counts(1)/sum(pair_class_counts));
            
            pair_weights(2, :) = (pair_S_inv * pair_class_means(:, 2))';
            pair_bias(2) = -0.5 * pair_class_means(:, 2)' * pair_S_inv * pair_class_means(:, 2) + log(pair_class_counts(2)/sum(pair_class_counts));
            
            % Store classifier
            pairwise_classifiers{a1, a2} = struct(...
                'weights', pair_weights, ...
                'bias', pair_bias, ...
                'angles', [a1, a2]);
        end
    end
    
    % Create specialized classifiers for problematic angles
    specialized_classifiers = cell(length(problem_angles), 1);

    for pa_idx = 1:length(problem_angles)
        pa = problem_angles(pa_idx);
        
        % Binary classification: angle vs others
        binary_y = double(Y_class == pa);
        
        % Logistic regression
        % Center the features
        X_mean = mean(X_class, 2);
        X_centered = X_class - X_mean;
        
        reg_param = 0.1;
        XX_t = X_centered * X_centered';
        reg_matrix = reg_param * eye(size(XX_t, 1));
        binary_weights = inv(XX_t + reg_matrix) * (X_centered * binary_y');
        
        % Calculate bias term
        binary_bias = mean(binary_y) - X_mean' * binary_weights;
        
        specialized_classifiers{pa_idx} = struct(...
            'angle', pa, ...
            'weights', binary_weights, ...
            'bias', binary_bias);
    end
    
    % Create and analyze confusion matrix
    confusion_weights = zeros(num_angles, num_angles);
    for a1 = 1:num_angles
        for a2 = 1:num_angles
            if a1 ~= a2
                % Find samples from angle a1 that get classified as a2
                idx_a1 = Y_class == a1;
                if sum(idx_a1) > 0
                    scores_a1 = lda_weights * X_class(:, idx_a1) + lda_bias;
                    [~, pred_classes] = max(scores_a1);
                    confusion_weights(a1, a2) = sum(pred_classes == a2) / sum(idx_a1);
                end
            end
        end
    end

    % Package model parameters
    modelParameters.avg_trajectories = avg_trajectories;
    modelParameters.lda_weights = lda_weights;
    modelParameters.lda_bias = lda_bias;
    modelParameters.prep_window = prep_window;
    modelParameters.bin_size = bin_size;
    modelParameters.window_size = window_size;
    modelParameters.max_bins = max_bins;

    modelParameters.pairwise_classifiers = pairwise_classifiers;
    modelParameters.confidence_margin_threshold = angle_confidence_threshold;
    
    modelParameters.pca_components = pca_components;
    modelParameters.explained_variance = explained(1:num_pca_components);
    
    % Velocity parameters
    % modelParameters.vel_beta = beta;
    % modelParameters.X_vel_mean = X_vel_mean;
    % modelParameters.Y_vel_mean = Y_vel_mean;
    
    % Specialized classifier parameters
    modelParameters.specialized_classifiers = specialized_classifiers;
    modelParameters.problem_angles = problem_angles;
    modelParameters.confusion_weights = confusion_weights;

    modelParameters.num_pca_components = num_pca_components;
end