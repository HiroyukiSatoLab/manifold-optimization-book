% 問題の作成
n = 2000; % 対称行列 A のサイズ
rng(0);
D = diag(1+.01*(1:n)); % A の固有値を並べた対角行列
[Q, ~] = qr(randn(n)); % A の固有ベクトルを並べた直交行列
A = Q*D*Q'; % 正定値対称行列 A を生成
b = randn(n,1); % ベクトル b を生成

% 目的関数の定義域である多様体をユークリッド空間 R^n と設定
manifold = euclideanfactory(n);
problem = [];
problem.M = manifold;

% 目的関数とその勾配およびヘッセ行列（のベクトルとの積）を定義
problem.cost  = @(x) 0.5 * x'*(A*x) - b'*x;
problem.egrad = @(x) A*x - b;
problem.ehess = @(x, d) A*d;

% 初期点と停止条件の設定
x0 = randn(n,1);
options = [];
options.tolgradnorm = 1e-6; % 目的関数の勾配のノルムがこの値より小さくなったら停止

% 種々のアルゴリズムで問題を解く
[xSD, costSD, infoSD] = steepestdescent(problem, x0, options); % 最急降下法
[xCG, costCG, infoCG] = conjugategradient(problem, x0, options); % 共役勾配法
[xBFGS, costBFGS, infoBFGS] = rlbfgs(problem, x0, options); % 準ニュートン法
[xTR, costTR, infoTR] = trustregions(problem, x0, options); % 信頼領域法

% 収束履歴の図示
figure;
h = semilogy([infoSD.iter], [infoSD.gradnorm], '.-', [infoCG.iter], [infoCG.gradnorm], '-x', [infoBFGS.iter], [infoBFGS.gradnorm], '-^', [infoTR.iter], [infoTR.gradnorm], '-*');
legend('最急降下法', '共役勾配法', '準ニュートン法', '信頼領域法', 'FontSize', 12);
xlabel('$$k$$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$$\|\mathrm{grad} f(x_k)\|_2$$', 'Interpreter', 'latex', 'FontSize', 16);
ylim([1e-7,1e+3*10^0.5]);

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
