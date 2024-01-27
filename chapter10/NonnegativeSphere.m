% 問題の作成
n = 5; % 対称行列 A のサイズ
rng(0);
A = rand(n)-.5;
A = A' + A;

% 目的関数の定義域である多様体を球面と設定
M = spherefactory(n);

% 目的関数とそのユークリッド勾配を定義
f = @(x) -x' * (A*x);
g = @(x) -x;
egradf = @(x) -2*A*x;
Dg = @(x) -eye(n); % g の微分（ヤコビ行列）

% 拡張ラグランジュ法のパラメータを設定
x0 = ones(n,1); x0 = x0 / norm(x0); % 球面上の初期点
mu0 = zeros(n,1);
epsilon0 = 0.001;
rho0 = 1;
OuterIter = 35; % 拡張ラグランジュ関数の反復回数
epsilonFinal = 1e-6; % 反復の最後での epsilon_k の値
theta_epsilon = nthroot(epsilonFinal/epsilon0, OuterIter-1);
theta_rho = 1.0 / 0.3;
theta_sigma = 0.8;
mu_max = 20 * ones(n,1);

% 拡張ラグランジュ法を適用
[xsol, time] = AugmentedLagrangian_Inequality(M, f, g, egradf, Dg, x0, mu0, epsilon0, rho0, OuterIter, theta_epsilon, theta_rho, theta_sigma, mu_max);
Ax = A * xsol; % 次の行でベクトル A * x を 2 回用いて計算するのであらかじめ計算しておく
KKT = Ax - (xsol'*Ax) * xsol; % (I_n - xx^T)Ax

fprintf('実行時間　　 ：%f秒\n', time);
fprintf('得られた解　 ：x = [%f; %f; %f; %f; %f]\n', xsol(1), xsol(2), xsol(3), xsol(4), xsol(5));
fprintf('KKT条件の確認：(I_n - xx^T)Ax = [%f; %f; %f; %f; %f]\n', KKT(1), KKT(2), KKT(3), KKT(4), KKT(5)); % この結果が 0 以下かつ x >= 0 ならば KKT 条件が成り立つ
