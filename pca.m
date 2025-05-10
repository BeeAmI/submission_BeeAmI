function pca()
    % Plot the PCA
    % Taken out of the training function for the model
    training_data = load('monkeydata_training.mat');
    training_data = training_data(1,1).trial;
    num_trials = size(training_data, 1);
    num_angles = size(training_data, 2);
    num_neurons = size(training_data(1,1).spikes, 1);

    bin_size = 20;
    prep_window = 300 + 20;
    window_size = 20;
    num_pca_components = 13;
    angle_confidence_threshold = 0.25;
    
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
        end
    end
    
    % Trim arrays to actual size
    all_firing_rates = all_firing_rates(:, 1:fr_idx-1);
    all_start_positions = all_start_positions(:, 1:fr_idx-1);
    Y_class = Y_class(1:fr_idx-1);

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

    % Plot explained variance vs principal components
    figure;
    plot(1:length(explained), explained, '-', 'LineWidth', 10, 'DisplayName','PCA Variance', 'Color', 'black');
    hold on;
    
    % Add vertical lines
    xline(13, '-.', 'Color', 'blue', 'Alpha', 1, 'LineWidth', 10, 'DisplayName', 'Chosen Threshold');
    xline(17, ':', 'Color', '#ff6633', 'Alpha', 1, 'LineWidth', 10, 'DisplayName', 'Threshold at 70%');
    set(gca, 'FontSize', 48);

    % Labels and formatting
    xlabel('Principal Components', 'FontSize', 48);
    ylabel('Explained Variance (%)', 'FontSize', 48);
    legend('Location', 'northeast', 'FontSize', 36);
    % grid on;
    box off;
    xlim([0, 40])
    ylim([-2.5,27.5]);
    yticks([0:5:25]);

end