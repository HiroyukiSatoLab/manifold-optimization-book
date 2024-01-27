% 多様体 M 上の点 p における接空間で定義された線形変換 A についての方程式 A(x) = b を解くための線形共役勾配法
% 初期点 x0 から反復を開始し，A(x) - b のノルムが tol 未満になったら計算を終了する．
% 計算終了時に得られている近似解を out, 計算終了までに要した反復回数を iter, 時間を time として出力する．
function [out, iter, time] = GeneralLinearCG(M, p, A, b, x0, tol)
tic; % 時間計測開始
x = x0; % 初期ベクトル
r = M.lincomb(p, 1, A(x), -1, b); % r = A(x) - b;
d = M.lincomb(p, -1, r); % d = -r;
rNorm2 = M.inner(p, r,r); % ||r||^2
iter = 0; % 反復回数

while sqrt(rNorm2) >= tol % ||r|| が tol 未満になったら計算を終了する
    Ad = A(d); % A(d) が必要な箇所が2つあるので，1回だけの計算で済むよう先に計算しておく
    t = rNorm2 / M.inner(p, d, Ad); % t = <r, r> / <d, Ad>;
    x = M.lincomb(p, 1, x, t, d); % x = x + t * d;
    rPrev = r; % 更新前の r を保存しておく
    rNorm2Prev = rNorm2; % 更新前の r について ||r||^2 を保存しておく
    r = M.lincomb(p, 1, r, t, Ad); % r = r + t * Ad;
    rNorm2 = M.inner(p, r, r); % 更新後の r について ||r||^2 を計算する
    beta = rNorm2 / M.inner(p, rPrev, rPrev); % beta = ||r_{k+1}||^2 / ||r_k||^2;
    d = M.lincomb(p, -1, r, beta, d); % d = -r + beta * d;
    iter = iter + 1; % 反復回数をカウント
end

out = x;
time = toc; % 時間計測終了

end
