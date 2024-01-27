% 問題の作成
m = 30; n = 20; p = 5;
rng(0);
A = rand(m,n);
N = diag(p:-1:1); % 重みを表す対角行列

% 目的関数の定義域である多様体をシュティーフェル多様体の積多様体と設定
M1 = stiefelfactory(m,p);
M2 = stiefelfactory(n,p);
problem = [];
problem.M = productmanifold(struct('U', M1, 'V', M2)); % M = M1 x M2

% 目的関数とそのユークリッド勾配を定義
problem.cost  = @(X) -trace(X.U' * A * X.V * N);
problem.egrad = @(X) struct('U', -A * X.V * N, 'V', -A' * X.U * N);
problem.ehess = @(X,D) struct('U', -A * D.V * N, 'V', -A' * D.U * N);

% 最適解を svd 関数により求め，その近くの点を最適化の初期点とする
[U, ~, V] = svd(A); % A の特異値分解
X0 = [];
X0.U = qr_unique(U(:,1:p) + .01 * (rand(m,p)-.5)); X0.V = qr_unique(V(:,1:p) + .01 * (rand(n,p)-.5)); % ノイズを加えてからQR分解することで最適解の近くの点 X0 を計算
X = X0;

%% ニュートン法を実行
fprintf('--------------------ニュートン法--------------------\n');
fprintf('反復番号　　勾配のノルム　　LCGの反復回数　　LCGの計算時間\n');
fprintf('--------------------------------------------------\n');
tNewton = tic; % ニュートン法の時間計測開始
EGrad = problem.egrad(X); % ユークリッド勾配
Grad = problem.M.egrad2rgrad(X, EGrad); % リーマン勾配

k = 0;
normGrad = problem.M.norm(X, Grad); % リーマン勾配のノルム
fprintf('　　%d　　　%.10f\n', k, normGrad);

while normGrad >= 1e-6
    k = k + 1;
    Hess = @(D) problem.M.ehess2rhess(X, EGrad, problem.ehess(X,D), D);
    NegativeGrad = problem.M.lincomb(X, -1, Grad); % -grad f(X) を計算
    [eta, iterCG(k), timeCG(k)] = GeneralLinearCG(problem.M, X, Hess, NegativeGrad, problem.M.zerovec(X), 1e-6); % ニュートン方程式を解く
    X = problem.M.retr(X, eta); % レトラクション R により R_X(eta) を計算し，次の点とする
    EGrad = problem.egrad(X);
    Grad = problem.M.egrad2rgrad(X, EGrad);
    normGrad = problem.M.norm(X, Grad);
    fprintf('　　%d　　　%.10f　　　%3d回　　　　%f秒\n', k, normGrad, iterCG(k), timeCG(k));
end

tNewton = toc(tNewton); % ニュートン法の時間計測終了

%% 共役勾配法を実行
fprintf('\n');
fprintf('--------------------共役勾配法--------------------\n');
[XCG, costCG, infoCG] = conjugategradient(problem, X0);

%% 両者を比較
fprintf('\n');
fprintf('--------------------両者の比較--------------------\n');
fprintf('　　　アルゴリズム　　　　　 反復回数　　　計算時間\n');
fprintf('------------------------------------------------\n');
fprintf('　　　ニュートン法　　　　　　%3d回　　　%f秒\n', k, tNewton);
fprintf('（多様体上の）共役勾配法　　　%3d回　　　%f秒\n', infoCG(end).iter, infoCG(end).time);
