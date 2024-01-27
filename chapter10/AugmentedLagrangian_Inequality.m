% リーマン多様体 M 上の目的関数 f, 不等式制約 g(x) <= 0 をもつ最適化問題に対する拡張ラグランジュ法
% 初期点 x0 と初期ラグランジュ乗数ベクトル mu0 から反復を開始し OuterIter 反復後に計算を終了する．
% 計算終了時に得られている近似解を out, 計算終了までに要した時間を time として出力する．
function [out, time] = AugmentedLagrangian_Inequality(M, f, g, egradf, Dg, x0, mu0, epsilon0, rho0, OuterIter, theta_epsilon, theta_rho, theta_sigma, mu_max)
tic; % 時間計測開始
x = x0; % 初期点
mu = mu0; % 初期ラグランジュ乗数ベクトル
epsilon = epsilon0; rho = rho0; % 初期パラメータ
Pi = @(x,a,b) max(a, min(x,b)); % 閉集合 [a_1, b_1] x ... x [a_m, b_m] への射影
L = @(x, mu, rho) f(x) + .5 * rho * norm(max(0, mu/rho + g(x)))^2; % 拡張ラグランジュ関数

problem = [];
problem.M = M;
options = [];
options.verbosity = 1;

for k = 0 : OuterIter-1 % OuterIter 回反復する
    % M 上の部分問題の定義
    problem.cost = @(x) L(x, mu, rho); % 各反復で最小化すべき関数 l_k
    problem.egrad = @(x) egradf(x) + rho * Dg(x)' * max(0, mu/rho + g(x));
    % 部分問題を準ニュートン法で解く
    fprintf('k = %d:\n', k);
    options.tolgradnorm = epsilon; % 部分問題は l_k の勾配のノルムが epsilon 未満になるように解く
    [x, ~, ~] = rlbfgs(problem,x,options);

    mu = Pi(mu + rho*g(x), 0, mu_max); % mu_k の更新
    if k >= 1; sigmaPrev = sigma; end % 更新前の sigma を保存
    sigma = max(g(x), -mu/rho); % sigma_k の更新
    epsilon = theta_epsilon * epsilon; % epsilon_k の更新
    if k >= 1 && max(sigma) > theta_sigma * max(sigmaPrev)
        rho = theta_rho * rho; % rho_k の更新
    end
end

out = x;
time = toc; % 時間計測終了

end
