% 3つの正規分布の平均
mu1 = [0; 0];
mu2 = [-2; 2];
mu3 = [2; 2];

% 3つの正規分布の分散
Sigma1 = [1.5 -0.2; -0.2 0.1];
Sigma2 = [1 0; 0 0.5];
Sigma3 = [0.8 0.3; 0.3 1];

% 3つの正規分布それぞれに従うデータ点を100個, 200個, 300個生成
rng(5);
R1 = mvnrnd(mu1, Sigma1, 200);
R2 = mvnrnd(mu2, Sigma2, 300);
R3 = mvnrnd(mu3, Sigma3, 500);

% それぞれのデータの散布図を描く（図9.4）
figure; hold on;
plot(R1(:, 1), R1(:, 2), '*');
plot(R2(:, 1), R2(:, 2), '_');
plot(R3(:, 1), R3(:, 2), '+');
legend('データ1', 'データ2', 'データ3');

% 実際に得られるデータはどの分布から得られたものかわからない（図9.5）
R = [R1; R2; R3];
figure; hold on;
plot(R(:, 1), R(:, 2), 'o');

% 目的関数の定義域である多様体を設定
Simplex = multinomialfactory(3); % 重みベクトルが属する多様体 Delta_3
Euclidean = euclideanfactory(2, 1); % 各平均ベクトルが属するユークリッド空間（平面） R^2
SymPositive = sympositivedefinitefactory(2); % 各分散共分散行列が属する多様体 Sym_{++}(2)
M1 = Simplex;
M2 = powermanifold(Euclidean, 3); % 3つの Euclidean の積多様体
M3 = powermanifold(SymPositive, 3); % 3つの SymPositive の積多様体
problem = [];
problem.M = productmanifold(struct('w', M1, 'mu', M2, 'Sigma', M3)); % M1, M2, M3 の積多様体

% 目的関数とその勾配を設定
problem.cost  = @(X) loglike(R, X.w, X.mu{1}, X.mu{2}, X.mu{3}, X.Sigma{1}, X.Sigma{2}, X.Sigma{3});
problem.grad = @(X) rgrad_loglike(R, X.w, X.mu{1}, X.mu{2}, X.mu{3}, X.Sigma{1}, X.Sigma{2}, X.Sigma{3}); % リーマン勾配
% problem.grad を指定する代わりに, 以下のようにユークリッド勾配を指定することも可能
% problem.egrad = @(X) egrad_loglike(R, X.w, X.mu{1}, X.mu{2}, X.mu{3}, X.Sigma{1}, X.Sigma{2}, X.Sigma{3}); % ユークリッド勾配

rng(0);
[X, Xcost, info, options] = conjugategradient(problem);

fprintf('--------------------------------------------------\n');
fprintf('推定された重み　　　　　 ：[%f; %f; %f]\n', X.w(1), X.w(2),X.w(3));
fprintf('推定された平均ベクトル1　：[%f; %f]\n', X.mu{1}(1), X.mu{1}(2));
fprintf('推定された平均ベクトル2　：[%f; %f]\n', X.mu{2}(1), X.mu{2}(2));
fprintf('推定された平均ベクトル3　：[%f; %f]\n', X.mu{3}(1), X.mu{3}(2));
fprintf('推定された分散共分散行列1：[%f %f; %f %f]\n', X.Sigma{1}(1,1), X.Sigma{1}(1,2), X.Sigma{1}(2,1), X.Sigma{1}(2,2));
fprintf('推定された分散共分散行列2：[%f %f; %f %f]\n', X.Sigma{2}(1,1), X.Sigma{2}(1,2), X.Sigma{2}(2,1), X.Sigma{2}(2,2));
fprintf('推定された分散共分散行列3：[%f %f; %f %f]\n', X.Sigma{3}(1,1), X.Sigma{3}(1,2), X.Sigma{3}(2,1), X.Sigma{3}(2,2));


% 平均ベクトル mu，分散共分散行列 Sigma の多変数正規分布の確率密度関数
function out = pdf(x, mu, Sigma)
out = exp(-0.5 * (x-mu)' * (Sigma \ (x-mu))) / sqrt(det(Sigma));
end

% 3つの多変数正規分布の確率密度関数の重みベクトル w についての和
function out = sumpdf(x, w, mu1, mu2, mu3, Sigma1, Sigma2, Sigma3)
out = w(1) * pdf(x, mu1, Sigma1) + w(2) * pdf(x, mu2, Sigma2) + w(3) * pdf(x, mu3, Sigma3);
end

% 対数尤度関数（R は サンプルサイズ x 2 のデータ）
function out = loglike(R, w, mu1, mu2, mu3, Sigma1, Sigma2, Sigma3)
[T, ~] = size(R); % T はサンプルサイズ
out = 0;
for t = 1 : T
    out = out + log(sumpdf([R(t,1); R(t,2)], w, mu1, mu2, mu3, Sigma1, Sigma2, Sigma3));
end
out = -out;
end

% 対数尤度関数のリーマン勾配（補足資料のIII.5節を参照）
function out = rgrad_loglike(R, w, mu1, mu2, mu3, Sigma1, Sigma2, Sigma3)
[T, ~] = size(R); % T はサンプルサイズ
sqrtdetS1 = sqrt(det(Sigma1)); sqrtdetS2 = sqrt(det(Sigma2)); sqrtdetS3 = sqrt(det(Sigma3)); % 計算途中で複数回現れるのであらかじめ計算
out.w = zeros(3,1);
out.mu{1} = zeros(2,1);
out.mu{2} = zeros(2,1);
out.mu{3} = zeros(2,1);
out.Sigma{1} = zeros(2);
out.Sigma{2} = zeros(2);
out.Sigma{3} = zeros(2);

for t = 1:T
    xt = R(t,1:2)';
    sumt = sumpdf(xt, w, mu1, mu2, mu3, Sigma1, Sigma2, Sigma3);
    % 計算途中で複数回現れる量をあらかじめ計算する
    xm1 = xt - mu1; xm2 = xt - mu2; xm3 = xt - mu3;
    Sxm1 = Sigma1 \ xm1; Sxm2 = Sigma2 \ xm2; Sxm3 = Sigma3 \ xm3;
    exp1 = exp(-0.5 * xm1' * Sxm1); exp2 = exp(-0.5 * xm2' * Sxm2); exp3 = exp(-0.5 * xm3' * Sxm3);
    expSD1sumt = (exp1 / sqrtdetS1) /sumt; expSD2sumt = (exp2 / sqrtdetS2) /sumt; expSD3sumt = (exp3 / sqrtdetS3) /sumt;
    z1 = w(1) * expSD1sumt; z2 = w(2) * expSD2sumt; z3 = w(3) * expSD3sumt;
    % リーマン勾配の計算
    out.w = out.w - [expSD1sumt; expSD2sumt; expSD3sumt];
    out.mu{1} = out.mu{1} - z1 * Sxm1;
    out.mu{2} = out.mu{2} - z2 * Sxm2;
    out.mu{3} = out.mu{3} - z3 * Sxm3;
    out.Sigma{1} = out.Sigma{1} - 0.5 * z1 * (xm1 * xm1' - Sigma1);
    out.Sigma{2} = out.Sigma{2} - 0.5 * z2 * (xm2 * xm2' - Sigma2);
    out.Sigma{3} = out.Sigma{3} - 0.5 * z3 * (xm3 * xm3' - Sigma3);
end
% この時点での out.w はユークリッド勾配
out.w = (out.w + T) .* w; % 射影によるリーマン勾配の計算
end


% 目的関数のリーマン勾配ではなくユークリッド勾配を指定する場合（補足資料のIII.5節を参照）
% function out = egrad_loglike(R, w, mu1, mu2, mu3, Sigma1, Sigma2, Sigma3)
% [T, ~] = size(R); % T はサンプルサイズ
% sqrtdetS1 = sqrt(det(Sigma1)); sqrtdetS2 = sqrt(det(Sigma2)); sqrtdetS3 = sqrt(det(Sigma3)); % 計算途中で複数回現れるのであらかじめ計算
% out.w = zeros(3,1);
% out.mu{1} = zeros(2,1);
% out.mu{2} = zeros(2,1);
% out.mu{3} = zeros(2,1);
% out.Sigma{1} = zeros(2);
% out.Sigma{2} = zeros(2);
% out.Sigma{3} = zeros(2);
% 
% for t = 1:T
%     xt = R(t,1:2)';
%     sumt = sumpdf(xt, w, mu1, mu2, mu3, Sigma1, Sigma2, Sigma3);
%     % 計算途中で複数回現れる量をあらかじめ計算する
%     xm1 = xt - mu1; xm2 = xt - mu2; xm3 = xt - mu3;
%     Sxm1 = Sigma1 \ xm1; Sxm2 = Sigma2 \ xm2; Sxm3 = Sigma3 \ xm3;
%     exp1 = exp(-0.5 * xm1' * Sxm1); exp2 = exp(-0.5 * xm2' * Sxm2); exp3 = exp(-0.5 * xm3' * Sxm3);
%     expSD1sumt = (exp1 / sqrtdetS1) /sumt; expSD2sumt = (exp2 / sqrtdetS2) /sumt; expSD3sumt = (exp3 / sqrtdetS3) /sumt;
%     z1 = w(1) * expSD1sumt; z2 = w(2) * expSD2sumt; z3 = w(3) * expSD3sumt;
%     % ユークリッド勾配の計算
%     out.w = out.w - [expSD1sumt; expSD2sumt; expSD3sumt];
%     out.mu{1} = out.mu{1} - z1 * Sxm1;
%     out.mu{2} = out.mu{2} - z2 * Sxm2;
%     out.mu{3} = out.mu{3} - z3 * Sxm3;
%     out.Sigma{1} = out.Sigma{1} - 0.5 * z1 * (Sxm1 * Sxm1' - inv(Sigma1));
%     out.Sigma{2} = out.Sigma{2} - 0.5 * z2 * (Sxm2 * Sxm2' - inv(Sigma2));
%     out.Sigma{3} = out.Sigma{3} - 0.5 * z3 * (Sxm3 * Sxm3' - inv(Sigma3));
% end
% end
