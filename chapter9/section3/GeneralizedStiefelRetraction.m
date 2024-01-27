p = 10;

for n = 2000:2000:10000
    fprintf('n = %d\n', n);
    rng(0);
    G = randn(n); G = G' * G; % 正定値対称行列 G を生成

    % 多様体 M を一般化シュティーフェル多様体と設定
    M = stiefelgeneralizedfactory(n,p,G);

    % 多様体上の点 X および X での接ベクトル eta をランダムに生成
    X = M.rand();
    eta = M.randvec(X);

    % レトラクションの計算方法その1（(9.24)を素朴に計算）
    tic;
    for t = 1:1
        sqrtG = sqrtm(G);
        R1 = sqrtG \ qr_unique(sqrtG * (X+eta));
    end
    t1 = toc;
    fprintf('方法1（式(9.24)）: %f秒\n', t1);

    % レトラクションの計算方法その2（コレスキー分解に基づく(9.25)を計算）
    tic;
    for t = 1:1000
        Xeta = X + eta;
        XGX = Xeta' * G * Xeta;
        R = chol(XGX);
        R2 = Xeta / R;
    end
    t2 = toc;
    t2 = t2 / 1000; % 平均時間の計算
    fprintf('方法2（式(9.25)）: %f秒\n', t2);
end
