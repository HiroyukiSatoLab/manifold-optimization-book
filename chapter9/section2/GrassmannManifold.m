% 問題の作成
n = 300; p = 5;
rng(0);
A = rand(n)-.5;
A = A' + A;

% 目的関数の定義域である多様体をグラスマン多様体と設定
problem = [];
problem.M = grassmannfactory(n,p);

% 目的関数とそのユークリッド勾配を定義
problem.cost  = @(X) sum(sum(X .* (A*X))); % tr(X^T AX)
problem.egrad = @(X) 2 * A * X;
problem.ehess = @(X, D) 2 * A * D;

% 初期点と停止条件の設定
X0 = problem.M.rand();
options = [];
options.tolgradnorm = 1e-6; % 目的関数の勾配のノルムがこの値より小さくなったら停止
options.maxiter = 2000; % 反復回数がこの値に達したら停止

% 種々のアルゴリズムで解く
[xSD, costSD, infoSD] = steepestdescent(problem, X0, options); % 最急降下法
[xCG, costCG, infoCG] = conjugategradient(problem, X0, options); % 共役勾配法
[xBFGS, costBFGS, infoBFGS] = rlbfgs(problem, X0, options); % 準ニュートン法
[xTR, costTR, infoTR] = trustregions(problem, X0, options); % 信頼領域法

% 収束履歴の図示
figure;
h = semilogy([infoSD(1:min(end,200)).iter], [infoSD(1:min(end,200)).gradnorm], '.-', [infoCG.iter], [infoCG.gradnorm], '-x', [infoBFGS.iter], [infoBFGS.gradnorm], '-^', [infoTR.iter], [infoTR.gradnorm], '-*');
legend('最急降下法', '共役勾配法', '準ニュートン法', '信頼領域法', 'FontSize', 12);
xlabel('$$k$$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$$\|\mathrm{grad} f(x_k)\|_2$$', 'Interpreter', 'latex', 'FontSize', 16);

MS = 8;
h(1).MarkerSize = MS;
h(2).MarkerSize = MS;
h(3).MarkerSize = MS;
h(4).MarkerSize = MS;

% 結果の表示
fprintf('--------------------------------------------------\n');
fprintf('アルゴリズム　　反復回数　　計算時間　　1反復あたりの計算時間\n');
fprintf('--------------------------------------------------\n');
fprintf('最急降下法　　　 %3d回　　%f秒　　%f秒\n', infoSD(end).iter, infoSD(end).time, infoSD(end).time/infoSD(end).iter);
fprintf('共役勾配法　　　 %3d回　　%f秒　　%f秒\n', infoCG(end).iter, infoCG(end).time, infoCG(end).time/infoCG(end).iter);
fprintf('準ニュートン法　 %3d回　　%f秒　　%f秒\n', infoBFGS(end).iter, infoBFGS(end).time, infoBFGS(end).time/infoBFGS(end).iter);
fprintf('信頼領域法　　　 %3d回　　%f秒　　%f秒\n', infoTR(end).iter, infoTR(end).time, infoTR(end).time/infoTR(end).iter);
