% 藍芽定位系統 GMEE-AEKF 模擬（採用 GPS GMEE 適應性演算法）
% Bluetooth Localization with Adaptive GMEE-EKF + LS baseline

clear all; close all; clc;

%% 參數設置
n_components = 3;  % 高斯混合分量數
state_dim = 4;     % 狀態維度 [x, y, vx, vy]
dt = 0.1;          % 時間步長
tx_power = -59;    % 藍芽發射功率 (dBm)
noise_std = 3.0;   % RSSI 測量雜訊標準差
path_loss_exp = 2.5;  % 路徑損耗指數

%% 設置藍芽信標位置
beacons = [
    0, 0;
    10, 0;
    10, 10;
    0, 10
];
n_beacons = size(beacons, 1);

%% 生成真實軌跡（圓形運動）
t = linspace(0, 4*pi, 200);
true_trajectory = [5 + 3*cos(t)', 5 + 3*sin(t)'];
n_steps = size(true_trajectory, 1);

%% 初始化 GMEE-AEKF
gmee_ekf = initialize_gmee_ekf(n_components, state_dim, dt);
gmee_ekf.means(1, :) = [5, 5, 0, 0];  % 初始位置設置

%% 存儲結果
estimates = zeros(n_steps, 2);          % GMEE-AEKF 位置估計
ls_estimates = zeros(n_steps, 2);       % LS 位置估計
measurements_history = zeros(n_steps, n_beacons);
errors = zeros(n_steps, 1);             % GMEE errors
errors_ls = zeros(n_steps, 1);          % LS errors
alpha_history_all = cell(n_steps, 1);   % 每步的 alpha 歷史
beta_history_all = cell(n_steps, 1);    % 每步的 beta 歷史

%% 主要模擬循環
fprintf('開始模擬藍芽定位系統（採用適應性 GMEE-EKF）...\n');
for k = 1:n_steps
    if mod(k, 50) == 0
        fprintf('進度: %d/%d\n', k, n_steps);
    end
    
    true_pos = true_trajectory(k, :);
    
    % 生成 RSSI 測量
    measurements = zeros(n_beacons, 1);
    for i = 1:n_beacons
        distance = norm(beacons(i, :) - true_pos);
        rssi = tx_power - 10 * path_loss_exp * log10(max(distance, 0.1));
        rssi = rssi + noise_std * randn();
        measurements(i) = rssi;
    end
    measurements_history(k, :) = measurements';
    
    %% ----- Least Squares (LS) 定位（baseline） -----
    est_d = 10.^((tx_power - measurements) ./ (10 * path_loss_exp));
    
    if n_beacons >= 3
        A = zeros(n_beacons-1, 2);
        b = zeros(n_beacons-1, 1);
        x1 = beacons(1,1); y1 = beacons(1,2); d1 = est_d(1);
        for j = 2:n_beacons
            xj = beacons(j,1); yj = beacons(j,2); dj = est_d(j);
            A(j-1, :) = [2*(xj - x1), 2*(yj - y1)];
            b(j-1) = (xj^2 - x1^2) + (yj^2 - y1^2) + (d1^2 - dj^2);
        end
        if rank(A) >= 2
            p_ls = A \ b;
        else
            p_ls = pinv(A) * b;
        end
        ls_estimates(k, :) = p_ls';
    else
        ls_estimates(k, :) = [NaN, NaN];
    end
    
    %% ----- GMEE-AEKF 預測與更新 -----
    gmee_ekf = predict_gmee_ekf(gmee_ekf);
    [gmee_ekf, alpha_hist, beta_hist] = update_gmee_aekf(gmee_ekf, measurements, beacons, tx_power, path_loss_exp);
    
    % 記錄參數歷史
    alpha_history_all{k} = alpha_hist;
    beta_history_all{k} = beta_hist;
    
    estimate = get_estimate(gmee_ekf);
    estimates(k, :) = estimate(1:2);
    
    %% 計算誤差
    errors(k) = norm(estimates(k, :) - true_pos);
    errors_ls(k) = norm(ls_estimates(k, :) - true_pos);
end

%% 繪製結果
fprintf('繪製結果...\n');
figure('Position', [100, 100, 1600, 1200]);

% 子圖1: 軌跡追蹤
subplot(3, 2, 1);
plot(true_trajectory(:, 1), true_trajectory(:, 2), 'b-', 'LineWidth', 2, 'DisplayName', '真實軌跡');
hold on;
plot(estimates(:, 1), estimates(:, 2), 'r-', 'LineWidth', 2, 'DisplayName', 'GMEE-AEKF估計');
plot(ls_estimates(:,1), ls_estimates(:,2), 'b-.', 'LineWidth', 1.5, 'DisplayName', 'LS估計');
plot(beacons(:, 1), beacons(:, 2), 'g^', 'MarkerSize', 15, 'LineWidth', 2, 'DisplayName', '藍芽信標');
xlabel('X 位置 (m)', 'FontSize', 12);
ylabel('Y 位置 (m)', 'FontSize', 12);
title('藍芽定位軌跡追蹤（GMEE-AEKF vs LS）', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;
axis equal;

% 子圖2: 位置誤差比較
subplot(3, 2, 2);
hold on;
plot(errors, 'r-', 'LineWidth', 2, 'DisplayName', 'GMEE-AEKF 誤差');
plot(errors_ls, 'b-', 'LineWidth', 1.5, 'DisplayName', 'LS 誤差');
xlabel('時間步', 'FontSize', 12);
ylabel('位置誤差 (m)', 'FontSize', 12);
title(sprintf('定位誤差 (GMEE mean: %.2fm, std: %.2fm)  (LS mean: %.2fm, std: %.2fm)', ...
    mean(errors), std(errors), mean(errors_ls,'omitnan'), std(errors_ls,'omitnan')), ...
    'FontSize', 11, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% 子圖3: RSSI信號
subplot(3, 2, 3);
hold on;
colors = lines(n_beacons);
for i = 1:n_beacons
    plot(measurements_history(:, i), 'Color', colors(i, :), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('信標 %d', i));
end
xlabel('時間步', 'FontSize', 12);
ylabel('RSSI (dBm)', 'FontSize', 12);
title('藍芽RSSI信號', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% 子圖4: Alpha 參數演變（選擇幾個代表性時間步）
subplot(3, 2, 4);
hold on;
sample_steps = [1, 50, 100, 150, 200];
colors_alpha = lines(length(sample_steps));
for idx = 1:length(sample_steps)
    step = sample_steps(idx);
    if step <= n_steps && ~isempty(alpha_history_all{step})
        plot(alpha_history_all{step}, 'Color', colors_alpha(idx,:), ...
            'LineWidth', 1.5, 'DisplayName', sprintf('步驟 %d', step));
    end
end
xlabel('迭代次數', 'FontSize', 12);
ylabel('Alpha 參數', 'FontSize', 12);
title('Alpha 參數在不同時間步的演變', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;

% 子圖5: Beta 參數演變
subplot(3, 2, 5);
hold on;
for idx = 1:length(sample_steps)
    step = sample_steps(idx);
    if step <= n_steps && ~isempty(beta_history_all{step})
        plot(beta_history_all{step}, 'Color', colors_alpha(idx,:), ...
            'LineWidth', 1.5, 'DisplayName', sprintf('步驟 %d', step));
    end
end
xlabel('迭代次數', 'FontSize', 12);
ylabel('Beta 參數', 'FontSize', 12);
title('Beta 參數在不同時間步的演變', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'best');
grid on;


%% 統計信息
fprintf('\n==================================================\n');
fprintf('藍芽定位系統 GMEE-AEKF 模擬結果（含 LS baseline）\n');
fprintf('==================================================\n');
fprintf('GMEE-AEKF:\n');
fprintf('  平均定位誤差: %.3f m\n', mean(errors));
fprintf('  誤差標準差: %.3f m\n', std(errors));
fprintf('  最大誤差: %.3f m\n', max(errors));
fprintf('  最小誤差: %.3f m\n', min(errors));
fprintf('\nLS (Least Squares):\n');
fprintf('  平均定位誤差: %.3f m\n', mean(errors_ls,'omitnan'));
fprintf('  誤差標準差: %.3f m\n', std(errors_ls,'omitnan'));
fprintf('  最大誤差: %.3f m\n', max(errors_ls));
fprintf('  最小誤差: %.3f m\n', min(errors_ls));
fprintf('\n高斯混合分量數: %d\n', n_components);
fprintf('最終權重分布 (GMEE-AEKF): ');
fprintf('%.3f ', gmee_ekf.weights);
fprintf('\n==================================================\n');

%% ==================== 函數定義 ====================

% 初始化 GMEE-EKF
function gmee_ekf = initialize_gmee_ekf(n_components, state_dim, dt)
    gmee_ekf.n_components = n_components;
    gmee_ekf.state_dim = state_dim;
    gmee_ekf.dt = dt;
    gmee_ekf.means = randn(n_components, state_dim) * 5;
    gmee_ekf.covariances = repmat(eye(state_dim) * 10, [1, 1, n_components]);
    gmee_ekf.weights = ones(n_components, 1) / n_components;
end

% 預測步驟
function gmee_ekf = predict_gmee_ekf(gmee_ekf)
    F = [
        1, 0, gmee_ekf.dt, 0;
        0, 1, 0, gmee_ekf.dt;
        0, 0, 1, 0;
        0, 0, 0, 1
    ];
    
    Q = eye(gmee_ekf.state_dim) * 0.5;
    
    for i = 1:gmee_ekf.n_components
        gmee_ekf.means(i, :) = (F * gmee_ekf.means(i, :)')';
        P = gmee_ekf.covariances(:, :, i);
        gmee_ekf.covariances(:, :, i) = F * P * F' + Q;
    end
end

% 適應性 GMEE-EKF 更新步驟（採用 GPS GMEE 演算法）
function [gmee_ekf, alpha_history, beta_history] = update_gmee_aekf(gmee_ekf, measurements, beacons, tx_power, path_loss_exp)
    n_beacons = size(beacons, 1);
    n_components = gmee_ekf.n_components;
    state_dim = gmee_ekf.state_dim;
    
    % 對每個高斯分量進行適應性更新
    for comp = 1:n_components
        % 提取當前分量的狀態
        xEst = gmee_ekf.means(comp, :)';
        PEst = gmee_ekf.covariances(:, :, comp);
        
        % 構建觀測模型
        predstat = xEst;
        tmppos = predstat(1:2)';  % [x, y]
        
        % 計算 H 矩陣（雅可比矩陣）
        H = zeros(n_beacons, state_dim);
        prped = zeros(n_beacons, 1);
        
        for i = 1:n_beacons
            beacon_pos = beacons(i, :);
            pred_distance = norm(beacon_pos - tmppos);
            
            % 預測 RSSI
            prped(i) = tx_power - 10 * path_loss_exp * log10(max(pred_distance, 0.1));
            
            % 計算雅可比矩陣
            if pred_distance > 0.1
                dx = tmppos(1) - beacon_pos(1);
                dy = tmppos(2) - beacon_pos(2);
                dh_dx = -10 * path_loss_exp / (pred_distance * log(10)) * dx / pred_distance;
                dh_dy = -10 * path_loss_exp / (pred_distance * log(10)) * dy / pred_distance;
                H(i, :) = [dh_dx, dh_dy, 0, 0];
            end
        end
        
        
        R = 9.0 * eye(n_beacons);
        z = measurements;
        
        % 構建聯合向量和協方差矩陣
        n = state_dim;
        m = n_beacons;
        B_all = [PEst, zeros(n, m); zeros(m, n), R];
        
        try
            B = chol(B_all, 'lower');
        catch
            B_all = B_all + 1e-8 * eye(size(B_all));
            B = chol(B_all, 'lower');
        end
        
        D = pinv(B) * [predstat; z];
        x3 = predstat;
        iter_count = 0;
        
        % 適應性參數設定
        alpha_base = 0.8;
        beta_base = 15.0;
        alpha_min = 0.3;
        alpha_max = 1.5;
        beta_min = 8.0;
        beta_max = 35.0;
        
        alpha = alpha_base;
        beta = beta_base;
        
        % 阻尼參數
        damping_factor = 0.6;
        
        % 收斂控制
        tol = 1e9;
        best_error = inf;
        best_x3 = x3;
        
        % 變化限制
        max_pos_change = 2.0;  % 藍芽系統位置變化較小
        min_iterations = 8;
        
        % 漸進式調整參數
        alpha_adjustment_rate = 0.06;
        beta_adjustment_rate = 0.2;
        error_threshold_low = 3.0;   % 藍芽系統誤差閾值較小
        error_threshold_high = 15.0;
        
        % 記錄歷史
        alpha_history = [];
        beta_history = [];
        error_history = [];
        
        % 主迭代循環
        max_iter = 20;
        while (tol >= 1e-6 || iter_count < min_iterations) && iter_count < max_iter
            x2 = x3;
            
            % 更新位置和 H 矩陣
            tmppos = x3(1:2)';
            for i = 1:n_beacons
                beacon_pos = beacons(i, :);
                pred_distance = norm(beacon_pos - tmppos);
                prped(i) = tx_power - 10 * path_loss_exp * log10(max(pred_distance, 0.1));
                
                if pred_distance > 0.1
                    dx = tmppos(1) - beacon_pos(1);
                    dy = tmppos(2) - beacon_pos(2);
                    dh_dx = -10 * path_loss_exp / (pred_distance * log(10)) * dx / pred_distance;
                    dh_dy = -10 * path_loss_exp / (pred_distance * log(10)) * dy / pred_distance;
                    H(i, :) = [dh_dx, dh_dy, 0, 0];
                else
                    H(i, :) = zeros(1, state_dim);
                end
            end
            
            % 計算殘差
            W = pinv(B) * [x3; prped];
            e = D - W;
            error_norm = norm(e);
            error_history(end+1) = error_norm;
            
            % 漸進式自適應參數調整
            if iter_count > 0
                % Alpha 調整
                if error_norm <= error_threshold_low
                    alpha_target = alpha_base + (alpha_max - alpha_base) * (error_threshold_low - error_norm) / error_threshold_low;
                elseif error_norm >= error_threshold_high
                    excess_error = error_norm - error_threshold_high;
                    reduction_factor = min(1.0, excess_error / error_threshold_high);
                    alpha_target = alpha_base - (alpha_base - alpha_min) * reduction_factor * 0.95;
                else
                    error_ratio = (error_norm - error_threshold_low) / (error_threshold_high - error_threshold_low);
                    alpha_target = alpha_base - (alpha_base - alpha_min) * error_ratio * 0.6;
                end
                
                % Beta 調整
                if error_norm <= error_threshold_low
                    beta_target = beta_min + (error_norm/error_threshold_low) * (beta_base - beta_min);
                elseif error_norm >= error_threshold_high
                    beta_target = beta_base + (error_norm - error_threshold_high) / error_threshold_high * (beta_max - beta_base);
                    beta_target = min(beta_target, beta_max);
                else
                    ratio = (error_norm - error_threshold_low) / (error_threshold_high - error_threshold_low);
                    beta_target = beta_base + ratio * (beta_max - beta_base) * 0.5;
                end
                
                % 漸進式更新
                max_alpha_step = alpha_adjustment_rate * (alpha_max - alpha_min);
                max_beta_step = beta_adjustment_rate * (beta_max - beta_min);
                
                alpha_diff = alpha_target - alpha;
                beta_diff = beta_target - beta;
                
                alpha_step = max(min(alpha_diff, max_alpha_step), -max_alpha_step);
                beta_step = max(min(beta_diff, max_beta_step), -max_beta_step);
                
                alpha = alpha + alpha_step;
                beta = beta + beta_step;
                
                alpha = max(alpha_min, min(alpha_max, alpha));
                beta = max(beta_min, min(beta_max, beta));
            end
            
            % 震盪檢測
            oscillation_detected = false;
            if length(error_history) >= 4
                recent_errors = error_history(end-3:end);
                if (recent_errors(2) > recent_errors(1) && recent_errors(3) < recent_errors(2) && recent_errors(4) > recent_errors(3)) || ...
                   (recent_errors(2) < recent_errors(1) && recent_errors(3) > recent_errors(2) && recent_errors(4) < recent_errors(3))
                    oscillation_detected = true;
                end
            end
            
            % 動態阻尼調整
            if oscillation_detected
                damping_factor = 0.7;
            else
                if length(error_history) >= 2
                    error_change_rate = abs(error_history(end) - error_history(end-1)) / (error_history(end-1) + 1e-6);
                    if error_change_rate > 0.3
                        damping_factor = min(0.8, damping_factor * 1.1);
                    elseif error_change_rate < 0.05
                        damping_factor = max(0.2, damping_factor * 0.95);
                    end
                end
            end
            
            % 記錄參數
            alpha_history(end+1) = alpha;
            beta_history(end+1) = beta;
            
            % 計算權重
            L = n + m;
            psi = zeros(L, 1);
            
            gamma_arg = max(1/alpha, 1e-6);
            gamma_val = gamma(gamma_arg);
            
            for jj = 1:L
                try
                    psi(jj) = (alpha / (2 * beta * gamma_val)) * exp(-abs(e(jj)/beta)^alpha);
                catch
                    psi(jj) = 1e-10;
                end
                psi(jj) = max(psi(jj), 1e-10);
            end
            
            psi = psi / sum(psi);
            
            % 構建熵矩陣
            Psi = diag(psi);
            C = Psi - psi * psi';
            
            % 正則化 C 矩陣
            try
                eig_vals = eig(C);
                min_eig = min(eig_vals);
                if min_eig < 0
                    C = C + (abs(min_eig) + 1e-8) * eye(size(C));
                end
            catch
                C = C + 1e-6 * eye(size(C));
            end
            
            % 協方差計算
            reg_factor = 1e-8;
            try
                B_n_inv = pinv(B(1:n, 1:n) + reg_factor * eye(n));
                B_m_inv = pinv(B(n+1:n+m, n+1:n+m) + reg_factor * eye(m));
                
                P_xx = B_n_inv' * C(1:n, 1:n) * B_n_inv;
                P_xy = B_m_inv' * C(n+1:n+m, 1:n) * B_n_inv;
                P_yx = B_n_inv' * C(1:n, n+1:n+m) * B_m_inv;
                R1 = B_m_inv' * C(n+1:n+m, n+1:n+m) * B_m_inv;
            catch
                P_xx = eye(n) * 1e-4;
                P_xy = zeros(m, n);
                P_yx = zeros(n, m);
                R1 = eye(m) * 1e-4;
            end
            
            % 計算卡爾曼增益
            try
                K_matrix = P_xx + H' * P_xy + (P_yx + H' * R1) * H + reg_factor * eye(n);
                K = K_matrix \ (P_yx + H' * R1);
            catch
                K = pinv(K_matrix) * (P_yx + H' * R1);
            end
            
            % 限制增益
            max_gain = 1.0;
            K(abs(K) > max_gain) = max_gain * sign(K(abs(K) > max_gain));
            
            % 計算創新量
            innovation = z - prped;
            max_innovation = 20;  % 藍芽系統創新量限制
            innovation = max(min(innovation, max_innovation), -max_innovation);
            
            % 阻尼更新
            x3_candidate = predstat + K * innovation;
            
            if any(isnan(x3_candidate)) || any(isinf(x3_candidate))
                x3 = 0.8 * x3 + 0.2 * predstat;
            else
                if iter_count > 0
                    x3 = (1 - damping_factor) * x3_candidate + damping_factor * x2;
                else
                    x3 = x3_candidate;
                end
                
                % 狀態變化檢查
                state_change = norm(x3 - x2);
                if state_change > max_pos_change
                    damping_factor = min(0.8, damping_factor * 1.2);
                end
                
                if error_norm < best_error
                    best_error = error_norm;
                    best_x3 = x3;
                end
            end
            
            % 收斂檢查
            tol = norm(x3 - x2) / (norm(x2) + 1e-10);
            iter_count = iter_count + 1;
        end
        
        % 使用最佳估計
        if best_error < inf
            x3 = best_x3;
        end
        
        % 更新高斯分量
        gmee_ekf.means(comp, :) = x3';
        
        % 協方差更新（Joseph 形式）
        Joseph_I = eye(size(K * H));
        PEst_updated = (Joseph_I - K * H) * PEst * (Joseph_I - K * H)' + K * R * K';
        
        % 確保正定
        try
            [V, D] = eig(PEst_updated);
            D_diag = diag(D);
            D_diag(D_diag < 1e-10) = 1e-10;
            PEst_updated = V * diag(D_diag) * V';
        catch
            PEst_updated = PEst_updated + 1e-8 * eye(size(PEst_updated));
        end
        
        gmee_ekf.covariances(:, :, comp) = PEst_updated;
    end
    
    % 更新權重（基於似然）
    new_weights = zeros(n_components, 1);
    for comp = 1:n_components
        likelihood = 0;
        tmppos = gmee_ekf.means(comp, 1:2);
        
        for i = 1:n_beacons
            beacon_pos = beacons(i, :);
            pred_distance = norm(beacon_pos - tmppos);
            pred_rssi = tx_power - 10 * path_loss_exp * log10(max(pred_distance, 0.1));
            innovation = measurements(i) - pred_rssi;
            
            R_scalar = 9.0;
            likelihood = likelihood - 0.5 * innovation^2 / R_scalar;
        end
        
        new_weights(comp) = gmee_ekf.weights(comp) * exp(likelihood);
    end
    
    % 歸一化權重
    total_weight = sum(new_weights);
    if total_weight > 0
        gmee_ekf.weights = new_weights / total_weight;
    end
end

% 獲取加權平均估計
function estimate = get_estimate(gmee_ekf)
    estimate = zeros(1, gmee_ekf.state_dim);
    for i = 1:gmee_ekf.n_components
        estimate = estimate + gmee_ekf.weights(i) * gmee_ekf.means(i, :);
    end
end