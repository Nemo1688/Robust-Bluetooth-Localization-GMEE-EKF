"""
藍芽定位系統 GMEE-AEKF 模擬
Bluetooth Localization with Adaptive GMEE-EKF + LS baseline
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, pinv
from scipy.special import gamma


plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


class GMEE_EKF:
    """GMEE-EKF 類別"""

    def __init__(self, n_components, state_dim, dt):
        self.n_components = n_components
        self.state_dim = state_dim
        self.dt = dt
        self.means = np.random.randn(n_components, state_dim) * 5
        self.covariances = np.tile(np.eye(state_dim) * 10, (n_components, 1, 1))
        self.weights = np.ones(n_components) / n_components

    def predict(self):
        """預測步驟"""
        F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        Q = np.eye(self.state_dim) * 0.5

        for i in range(self.n_components):
            self.means[i] = F @ self.means[i]
            P = self.covariances[i]
            self.covariances[i] = F @ P @ F.T + Q

    def update_aekf(self, measurements, beacons, tx_power, path_loss_exp):
        """ GMEE-EKF 更新步驟"""
        n_beacons = len(beacons)
        state_dim = self.state_dim
        alpha_history = []
        beta_history = []

        # 對高斯分量進行更新
        for comp in range(self.n_components):
            xEst = self.means[comp].copy()
            PEst = self.covariances[comp].copy()

            # 構建觀測模型
            predstat = xEst
            tmppos = predstat[:2]

            # 計算 H 矩陣和預測 RSSI
            H = np.zeros((n_beacons, state_dim))
            prped = np.zeros(n_beacons)

            for i in range(n_beacons):
                beacon_pos = beacons[i]
                pred_distance = np.linalg.norm(beacon_pos - tmppos)

                prped[i] = tx_power - 10 * path_loss_exp * np.log10(max(pred_distance, 0.1))

                if pred_distance > 0.1:
                    dx = tmppos[0] - beacon_pos[0]
                    dy = tmppos[1] - beacon_pos[1]
                    dh_dx = -10 * path_loss_exp / (pred_distance * np.log(10)) * dx / pred_distance
                    dh_dy = -10 * path_loss_exp / (pred_distance * np.log(10)) * dy / pred_distance
                    H[i] = [dh_dx, dh_dy, 0, 0]

            R = 9.0 * np.eye(n_beacons)
            z = measurements

            # 聯合向量和協方差矩陣
            n = state_dim
            m = n_beacons
            B_all = np.block([
                [PEst, np.zeros((n, m))],
                [np.zeros((m, n)), R]
            ])

            try:
                B = cholesky(B_all, lower=True)
            except:
                B_all = B_all + 1e-8 * np.eye(B_all.shape[0])
                B = cholesky(B_all, lower=True)

            D = pinv(B) @ np.concatenate([predstat, z])
            x3 = predstat.copy()
            iter_count = 0

            # 參數設定
            alpha_base = 0.8
            beta_base = 15.0
            alpha_min = 0.3
            alpha_max = 1.5
            beta_min = 8.0
            beta_max = 35.0

            alpha = alpha_base
            beta = beta_base
            damping_factor = 0.6

            tol = 1e9
            best_error = np.inf
            best_x3 = x3.copy()

            max_pos_change = 2.0
            min_iterations = 8

            alpha_adjustment_rate = 0.06
            beta_adjustment_rate = 0.2
            error_threshold_low = 3.0
            error_threshold_high = 15.0

            error_history = []
            alpha_hist = []
            beta_hist = []

            # 主循環
            max_iter = 20
            while (tol >= 1e-6 or iter_count < min_iterations) and iter_count < max_iter:
                x2 = x3.copy()

                # 更新位置和 H 矩陣
                tmppos = x3[:2]
                for i in range(n_beacons):
                    beacon_pos = beacons[i]
                    pred_distance = np.linalg.norm(beacon_pos - tmppos)
                    prped[i] = tx_power - 10 * path_loss_exp * np.log10(max(pred_distance, 0.1))

                    if pred_distance > 0.1:
                        dx = tmppos[0] - beacon_pos[0]
                        dy = tmppos[1] - beacon_pos[1]
                        dh_dx = -10 * path_loss_exp / (pred_distance * np.log(10)) * dx / pred_distance
                        dh_dy = -10 * path_loss_exp / (pred_distance * np.log(10)) * dy / pred_distance
                        H[i] = [dh_dx, dh_dy, 0, 0]
                    else:
                        H[i] = np.zeros(state_dim)

                # 計算殘差
                W = pinv(B) @ np.concatenate([x3, prped])
                e = D - W
                error_norm = np.linalg.norm(e)
                error_history.append(error_norm)

                # 漸進式自適應參數調整
                if iter_count > 0:
                    # Alpha 調整
                    if error_norm <= error_threshold_low:
                        alpha_target = alpha_base + (alpha_max - alpha_base) * (error_threshold_low - error_norm) / error_threshold_low
                    elif error_norm >= error_threshold_high:
                        excess_error = error_norm - error_threshold_high
                        reduction_factor = min(1.0, excess_error / error_threshold_high)
                        alpha_target = alpha_base - (alpha_base - alpha_min) * reduction_factor * 0.95
                    else:
                        error_ratio = (error_norm - error_threshold_low) / (error_threshold_high - error_threshold_low)
                        alpha_target = alpha_base - (alpha_base - alpha_min) * error_ratio * 0.6

                    # Beta 調整
                    if error_norm <= error_threshold_low:
                        beta_target = beta_min + (error_norm/error_threshold_low) * (beta_base - beta_min)
                    elif error_norm >= error_threshold_high:
                        beta_target = beta_base + (error_norm - error_threshold_high) / error_threshold_high * (beta_max - beta_base)
                        beta_target = min(beta_target, beta_max)
                    else:
                        ratio = (error_norm - error_threshold_low) / (error_threshold_high - error_threshold_low)
                        beta_target = beta_base + ratio * (beta_max - beta_base) * 0.5

                    # 漸進式更新
                    max_alpha_step = alpha_adjustment_rate * (alpha_max - alpha_min)
                    max_beta_step = beta_adjustment_rate * (beta_max - beta_min)

                    alpha_diff = alpha_target - alpha
                    beta_diff = beta_target - beta

                    alpha_step = np.clip(alpha_diff, -max_alpha_step, max_alpha_step)
                    beta_step = np.clip(beta_diff, -max_beta_step, max_beta_step)

                    alpha = alpha + alpha_step
                    beta = beta + beta_step

                    alpha = np.clip(alpha, alpha_min, alpha_max)
                    beta = np.clip(beta, beta_min, beta_max)

                # 震盪檢測
                oscillation_detected = False
                if len(error_history) >= 4:
                    recent = error_history[-4:]
                    if (recent[1] > recent[0] and recent[2] < recent[1] and recent[3] > recent[2]) or \
                       (recent[1] < recent[0] and recent[2] > recent[1] and recent[3] < recent[2]):
                        oscillation_detected = True

                # 動態阻尼調整
                if oscillation_detected:
                    damping_factor = 0.7
                else:
                    if len(error_history) >= 2:
                        error_change_rate = abs(error_history[-1] - error_history[-2]) / (error_history[-2] + 1e-6)
                        if error_change_rate > 0.3:
                            damping_factor = min(0.8, damping_factor * 1.1)
                        elif error_change_rate < 0.05:
                            damping_factor = max(0.2, damping_factor * 0.95)

                alpha_hist.append(alpha)
                beta_hist.append(beta)

                # 計算權重
                L = n + m
                psi = np.zeros(L)

                gamma_arg = max(1/alpha, 1e-6)
                gamma_val = gamma(gamma_arg)

                for jj in range(L):
                    try:
                        psi[jj] = (alpha / (2 * beta * gamma_val)) * np.exp(-abs(e[jj]/beta)**alpha)
                    except:
                        psi[jj] = 1e-10
                    psi[jj] = max(psi[jj], 1e-10)

                psi = psi / np.sum(psi)

                # 熵矩陣
                Psi = np.diag(psi)
                C = Psi - np.outer(psi, psi)

                # 正則化 C 矩陣
                try:
                    eig_vals = np.linalg.eigvals(C)
                    min_eig = np.min(eig_vals)
                    if min_eig < 0:
                        C = C + (abs(min_eig) + 1e-8) * np.eye(C.shape[0])
                except:
                    C = C + 1e-6 * np.eye(C.shape[0])

                # 協方差計算
                reg_factor = 1e-8
                try:
                    B_n_inv = pinv(B[:n, :n] + reg_factor * np.eye(n))
                    B_m_inv = pinv(B[n:n+m, n:n+m] + reg_factor * np.eye(m))

                    P_xx = B_n_inv.T @ C[:n, :n] @ B_n_inv
                    P_xy = B_m_inv.T @ C[n:n+m, :n] @ B_n_inv
                    P_yx = B_n_inv.T @ C[:n, n:n+m] @ B_m_inv
                    R1 = B_m_inv.T @ C[n:n+m, n:n+m] @ B_m_inv
                except:
                    P_xx = np.eye(n) * 1e-4
                    P_xy = np.zeros((m, n))
                    P_yx = np.zeros((n, m))
                    R1 = np.eye(m) * 1e-4

                # 計算卡爾曼增益
                try:
                    K_matrix = P_xx + H.T @ P_xy + (P_yx + H.T @ R1) @ H + reg_factor * np.eye(n)
                    K = np.linalg.solve(K_matrix, P_yx + H.T @ R1)
                except:
                    K = pinv(K_matrix) @ (P_yx + H.T @ R1)

                # 限制增益
                max_gain = 1.0
                K = np.clip(K, -max_gain, max_gain)

                # 計算創新量
                innovation = z - prped
                max_innovation = 20
                innovation = np.clip(innovation, -max_innovation, max_innovation)

                # 阻尼更新
                x3_candidate = predstat + K @ innovation

                if np.any(np.isnan(x3_candidate)) or np.any(np.isinf(x3_candidate)):
                    x3 = 0.8 * x3 + 0.2 * predstat
                else:
                    if iter_count > 0:
                        x3 = (1 - damping_factor) * x3_candidate + damping_factor * x2
                    else:
                        x3 = x3_candidate

                    state_change = np.linalg.norm(x3 - x2)
                    if state_change > max_pos_change:
                        damping_factor = min(0.8, damping_factor * 1.2)

                    if error_norm < best_error:
                        best_error = error_norm
                        best_x3 = x3.copy()

                tol = np.linalg.norm(x3 - x2) / (np.linalg.norm(x2) + 1e-10)
                iter_count += 1

            # 使用最佳估計
            if best_error < np.inf:
                x3 = best_x3

            # 更新高斯分量
            self.means[comp] = x3

            # 協方差更新（Joseph 形式）
            Joseph_I = np.eye(K.shape[0])
            PEst_updated = (Joseph_I - K @ H) @ PEst @ (Joseph_I - K @ H).T + K @ R @ K.T

            # 確保正定
            try:
                eigvals, eigvecs = np.linalg.eig(PEst_updated)
                eigvals = np.real(eigvals)
                eigvals[eigvals < 1e-10] = 1e-10
                PEst_updated = eigvecs @ np.diag(eigvals) @ eigvecs.T
                PEst_updated = np.real(PEst_updated)
            except:
                PEst_updated = PEst_updated + 1e-8 * np.eye(PEst_updated.shape[0])

            self.covariances[comp] = PEst_updated

            alpha_history.append(alpha_hist)
            beta_history.append(beta_hist)

        # 更新權重
        new_weights = np.zeros(self.n_components)
        for comp in range(self.n_components):
            likelihood = 0
            tmppos = self.means[comp, :2]

            for i in range(n_beacons):
                beacon_pos = beacons[i]
                pred_distance = np.linalg.norm(beacon_pos - tmppos)
                pred_rssi = tx_power - 10 * path_loss_exp * np.log10(max(pred_distance, 0.1))
                innovation = measurements[i] - pred_rssi

                R_scalar = 9.0
                likelihood += -0.5 * innovation**2 / R_scalar

            new_weights[comp] = self.weights[comp] * np.exp(likelihood)

        # 歸一化權重
        total_weight = np.sum(new_weights)
        if total_weight > 0:
            self.weights = new_weights / total_weight

        return alpha_history, beta_history

    def get_estimate(self):
        """獲取加權平均估計"""
        estimate = np.zeros(self.state_dim)
        for i in range(self.n_components):
            estimate += self.weights[i] * self.means[i]
        return estimate


def least_squares_localization(measurements, beacons, tx_power, path_loss_exp):
    """最小二乘法定位"""
    n_beacons = len(beacons)
    est_d = 10 ** ((tx_power - measurements) / (10 * path_loss_exp))

    if n_beacons >= 3:
        A = np.zeros((n_beacons - 1, 2))
        b = np.zeros(n_beacons - 1)
        x1, y1, d1 = beacons[0, 0], beacons[0, 1], est_d[0]

        for j in range(1, n_beacons):
            xj, yj, dj = beacons[j, 0], beacons[j, 1], est_d[j]
            A[j-1] = [2*(xj - x1), 2*(yj - y1)]
            b[j-1] = (xj**2 - x1**2) + (yj**2 - y1**2) + (d1**2 - dj**2)

        if np.linalg.matrix_rank(A) >= 2:
            p_ls = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            p_ls = pinv(A) @ b

        return p_ls
    else:
        return np.array([np.nan, np.nan])


def main():
    """主程式"""
    print("開始模擬藍芽定位...")

    # 參數設置
    n_components = 3
    state_dim = 4
    dt = 0.1
    tx_power = -59
    noise_std = 3.0
    path_loss_exp = 2.5

    # 設置藍芽信標位置
    beacons = np.array([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10]
    ])
    n_beacons = len(beacons)

    # 生成真實軌跡（圓形運動）
    t = np.linspace(0, 4*np.pi, 200)
    true_trajectory = np.column_stack([5 + 3*np.cos(t), 5 + 3*np.sin(t)])
    n_steps = len(true_trajectory)

    # 初始化 GMEE-AEKF
    gmee_ekf = GMEE_EKF(n_components, state_dim, dt)
    gmee_ekf.means[0] = [5, 5, 0, 0]

    # 存儲結果
    estimates = np.zeros((n_steps, 2))
    ls_estimates = np.zeros((n_steps, 2))
    measurements_history = np.zeros((n_steps, n_beacons))
    errors = np.zeros(n_steps)
    errors_ls = np.zeros(n_steps)

    # 主要模擬循環
    for k in range(n_steps):
        if (k + 1) % 50 == 0:
            print(f'進度: {k+1}/{n_steps}')

        true_pos = true_trajectory[k]

        # 生成 RSSI 測量
        measurements = np.zeros(n_beacons)
        for i in range(n_beacons):
            distance = np.linalg.norm(beacons[i] - true_pos)
            rssi = tx_power - 10 * path_loss_exp * np.log10(max(distance, 0.1))
            rssi = rssi + noise_std * np.random.randn()
            measurements[i] = rssi
        measurements_history[k] = measurements

        # Least Squares 定位
        ls_estimates[k] = least_squares_localization(measurements, beacons, tx_power, path_loss_exp)

        # GMEE-AEKF 預測與更新
        gmee_ekf.predict()
        alpha_hist, beta_hist = gmee_ekf.update_aekf(measurements, beacons, tx_power, path_loss_exp)

        estimate = gmee_ekf.get_estimate()
        estimates[k] = estimate[:2]

        # 計算誤差
        errors[k] = np.linalg.norm(estimates[k] - true_pos)
        errors_ls[k] = np.linalg.norm(ls_estimates[k] - true_pos)

    # 繪製結果
    print("繪製結果...")
    fig = plt.figure(figsize=(16, 12))

    # 子圖1: 軌跡追蹤
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(true_trajectory[:, 0], true_trajectory[:, 1], 'b-', linewidth=2, label='真實軌跡')
    ax1.plot(estimates[:, 0], estimates[:, 1], 'r-', linewidth=2, label='GMEE-AEKF估計')
    ax1.plot(ls_estimates[:, 0], ls_estimates[:, 1], 'b-.', linewidth=1.5, label='LS估計')
    ax1.plot(beacons[:, 0], beacons[:, 1], 'g^', markersize=15, linewidth=2, label='藍芽信標')
    ax1.set_xlabel('X 位置 (m)', fontsize=12)
    ax1.set_ylabel('Y 位置 (m)', fontsize=12)
    ax1.set_title('藍芽定位軌跡追蹤（GMEE-AEKF vs LS）', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True)
    ax1.axis('equal')

    # 子圖2: 位置誤差比較
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(errors, 'r-', linewidth=2, label='GMEE-AEKF 誤差')
    ax2.plot(errors_ls, 'b-', linewidth=1.5, label='LS 誤差')
    ax2.set_xlabel('時間步', fontsize=12)
    ax2.set_ylabel('位置誤差 (m)', fontsize=12)
    ax2.set_title(f'定位誤差 (GMEE mean: {np.mean(errors):.2f}m, std: {np.std(errors):.2f}m)  '
                  f'(LS mean: {np.nanmean(errors_ls):.2f}m, std: {np.nanstd(errors_ls):.2f}m)',
                  fontsize=11, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True)

    # 子圖3: RSSI信號
    ax3 = plt.subplot(3, 2, 3)
    for i in range(n_beacons):
        ax3.plot(measurements_history[:, i], linewidth=1.5, label=f'信標 {i+1}')
    ax3.set_xlabel('時間步', fontsize=12)
    ax3.set_ylabel('RSSI (dBm)', fontsize=12)
    ax3.set_title('藍芽RSSI信號', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # 統計信息
    print("\n" + "="*50)
    print("藍芽定位系統 GMEE-AEKF 模擬結果（含 LS baseline）")
    print("="*50)
    print("GMEE-AEKF:")
    print(f"  平均定位誤差: {np.mean(errors):.3f} m")
    print(f"  誤差標準差: {np.std(errors):.3f} m")
    print(f"  最大誤差: {np.max(errors):.3f} m")
    print(f"  最小誤差: {np.min(errors):.3f} m")
    print("\nLS (Least Squares):")
    print(f"  平均定位誤差: {np.nanmean(errors_ls):.3f} m")
    print(f"  誤差標準差: {np.nanstd(errors_ls):.3f} m")
    print(f"  最大誤差: {np.nanmax(errors_ls):.3f} m")
    print(f"  最小誤差: {np.nanmin(errors_ls):.3f} m")
    print(f"\n高斯混合分量數: {n_components}")
    print(f"最終權重分布 (GMEE-AEKF): {gmee_ekf.weights}")
    print("="*50)


if __name__ == "__main__":
    main()