
// sa https://qiita.com/ka10ryu1/items/bd05aed321a7a154d8a1
let imgMats = [];
let prevImgMats = [];
const cripTopValue = document.getElementById('cripTopValue');
const cripBottomValue = document.getElementById('cripBottomValue');
const cripRightValue = document.getElementById('cripRightValue');
const cripLeftValue = document.getElementById('cripLeftValue');

document.getElementById("load").addEventListener("click", () => {
    document.getElementById("imgs").click();
});

// クリップ値取得関数
function getClipValues() {
    return {
        top: parseInt(cripTopValue.value) || 0,
        bottom: parseInt(cripBottomValue.value) || 0,
        left: parseInt(cripLeftValue.value) || 0,
        right: parseInt(cripRightValue.value) || 0
    };
}

// プレビュー画像を再描画する関数
function updatePreview() {
    const scrollContent = document.getElementById('scroll_content');
    while (scrollContent.firstChild) {
        scrollContent.removeChild(scrollContent.firstChild);
    }
    prevImgMats.forEach((prevImgMat, idx) => {
        // imgMatsがまだロードされていない場合はスキップ
        if (!prevImgMat) return;

        // imgMatからcanvasへ描画
        let mat = prevImgMat;
        let canvas = document.createElement('canvas');
        canvas.width = mat.cols;
        canvas.height = mat.rows;
        cv.imshow(canvas, mat);

        // クリップ範囲に暗いマスクを描画
        let { top, bottom, left, right } = getClipValues();
        // プレビュー用に高さを1/3に調整
        top /= 3;
        bottom /= 3;
        left /= 3;
        right /= 3;
        let ctx = canvas.getContext('2d');
        ctx.save();
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = "#000";
        if (top > 0) ctx.fillRect(0, 0, canvas.width, top);
        if (bottom > 0) ctx.fillRect(0, canvas.height - bottom, canvas.width, bottom);
        if (left > 0) ctx.fillRect(0, top, left, canvas.height - top - bottom);
        if (right > 0) ctx.fillRect(canvas.width - right, top, right, canvas.height - top - bottom);
        ctx.restore();

        let previewImg = document.createElement('img');
        previewImg.style.maxWidth = "100%";
        previewImg.style.maxHeight = "100%";
        previewImg.style.margin = "4px";
        previewImg.src = canvas.toDataURL();
        scrollContent.appendChild(previewImg);
    });
}

function setProgress(percentage) {
    const progressBar = document.querySelector('#progress .progress-bar');
    progressBar.style.width = percentage + '%';
    progressBar.setAttribute('aria-valuenow', percentage);
}

document.getElementById('imgs').onchange = function (e) {
    let files = Array.from(e.target.files);
    files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));

    imgMats.forEach(mat => mat && mat.delete && mat.delete());
    imgMats = [];
    prevImgMats.forEach(mat => mat && mat.delete && mat.delete());
    prevImgMats = [];
    let loaded = 0;
    files.forEach((file, idx) => {
        let img = new Image();
        img.onload = function () {
            let canvas = document.createElement('canvas');
            canvas.width = img.width; canvas.height = img.height;
            canvas.getContext('2d').drawImage(img, 0, 0);
            imgMats[idx] = cv.imread(canvas);
            let prevCanvas = document.createElement('canvas');
            prevCanvas.width = Math.floor(img.width / 3);
            prevCanvas.height = Math.floor(img.height / 3);
            let ctx = prevCanvas.getContext('2d');
            ctx.drawImage(img, 0, 0, prevCanvas.width, prevCanvas.height);
            prevImgMats[idx] = cv.imread(prevCanvas);
            loaded++;
            if (loaded === files.length) {
                updatePreview();
            }
        };
        img.src = URL.createObjectURL(file);
    });
};

function stitch() {
    if (imgMats.length < 2) {
        alert("2枚以上の画像を選択してください");
        return;
    }
    let base = cripByPixels(imgMats[0], cripTopValue.value, cripBottomValue.value, cripLeftValue.value, cripRightValue.value).cripped;
    for (let i = 1; i < imgMats.length; i++) {
        let tgt_img = cripByPixels(imgMats[i], cripTopValue.value, cripBottomValue.value, cripLeftValue.value, cripRightValue.value);
        let result = stitchPairAffineAkaze(base, tgt_img.cripped);
        base.delete();
        base = result;
        // 進捗表示を更新
        setProgress(((i + 1) / imgMats.length) * 100.0);
    }
    cv.imshow('stitched_image', base);
    base.delete();
    document.getElementById('download').disabled = false;
    setProgress(0);
}

// mat画像を上下左右のピクセル数でクリップする
// cropTop, cropBottom, cropLeft, cropRight: それぞれ切り取るピクセル数
function cripByPixels(mat, cropTop = 0, cropBottom = 0, cropLeft = 0, cropRight = 0) {
    let w = mat.cols, h = mat.rows;
    let x = Math.max(0, cropLeft);
    let y = Math.max(0, cropTop);
    let cropW = Math.max(0, w - cropLeft - cropRight);
    let cropH = Math.max(0, h - cropTop - cropBottom);
    if (cropW <= 0 || cropH <= 0) {
        alert("クリップ範囲が無効です");
        // 切りすぎた場合は空画像を返す
        return { cropped: new cv.Mat(), offset: { x: 0, y: 0 } };
    }
    let rect = new cv.Rect(x, y, cropW, cropH);
    let cripped = mat.roi(rect);
    return { cripped, offset: { x, y } };
}

function stitchPairAffineAkaze(img1Mat, img2Mat) {
    // グレースケール変換
    let gray1 = new cv.Mat();
    let gray2 = new cv.Mat();
    cv.cvtColor(img1Mat, gray1, cv.COLOR_RGB2GRAY);
    cv.cvtColor(img2Mat, gray2, cv.COLOR_RGB2GRAY);

    // AKAZE特徴点抽出
    let akaze = new cv.AKAZE();
    let kp1 = new cv.KeyPointVector();
    let des1 = new cv.Mat();
    akaze.detectAndCompute(gray1, new cv.Mat(), kp1, des1);

    let kp2 = new cv.KeyPointVector();
    let des2 = new cv.Mat();
    akaze.detectAndCompute(gray2, new cv.Mat(), kp2, des2);

    let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
    let matches = new cv.DMatchVector();
    bf.match(des1, des2, matches);

    // マッチ点を距離でソート
    let matchesArr = [];
    for (let i = 0; i < matches.size(); i++) {
        matchesArr.push(matches.get(i));
    }
    matchesArr.sort((a, b) => a.distance - b.distance);

    // 最良の2組のマッチ点を使う
    if (matchesArr.length < 2) {
        alert("十分なマッチがありません");
        // メモリ解放
        gray1.delete(); gray2.delete(); akaze.delete(); kp1.delete(); kp2.delete();
        des1.delete(); des2.delete(); bf.delete(); matches.delete();
        return img1Mat.clone();
    }

    // 2組のマッチ点を使う。ただしスケールが0.8～1.2の範囲外なら次の組を探す
    let m0, m1, p1_0, p2_0, p1_1, p2_1, d1, d2, scale;
    let found = false;
    for (let i = 0; i < matchesArr.length - 1; i++) {
        m0 = matchesArr[i];
        m1 = matchesArr[i + 1];

        p1_0 = kp1.get(m0.queryIdx).pt;
        p2_0 = kp2.get(m0.trainIdx).pt;
        p1_1 = kp1.get(m1.queryIdx).pt;
        p2_1 = kp2.get(m1.trainIdx).pt;

        // スケール計算
        d1 = Math.hypot(p1_1.x - p1_0.x, p1_1.y - p1_0.y);
        d2 = Math.hypot(p2_1.x - p2_0.x, p2_1.y - p2_0.y);
        scale = d1 > 0 && d2 > 0 ? d1 / d2 : 1.0;

        if (scale >= 0.8 && scale <= 1.2) {
            found = true;
            break;
        }
    }
    if (!found) {
        alert("適切なスケールのマッチ点が見つかりません");
        gray1.delete(); gray2.delete(); akaze.delete(); kp1.delete(); kp2.delete();
        des1.delete(); des2.delete(); bf.delete(); matches.delete();
        img1Mat.delete(); img2Mat.delete();
        return img1Mat.clone();
    }

    // 平行移動計算（スケール適用後）
    let dx = p1_0.x - p2_0.x * scale;
    let dy = p1_0.y - p2_0.y * scale;

    // アフィン行列（回転なし、スケール＋平行移動のみ）
    let affineMat = cv.matFromArray(2, 3, cv.CV_64F, [
        scale, 0, dx,
        0, scale, dy
    ]);

    // 合成用画像
    let dsize = new cv.Size(img1Mat.cols + img2Mat.cols * 2, img1Mat.rows + img2Mat.rows * 2);
    let result = new cv.Mat.zeros(dsize.height, dsize.width, img1Mat.type());

    // img1Matを左側に貼る(img2のサイズ分オフセットした位置に貼る)
    let roi = result.roi(new cv.Rect(img2Mat.cols, img2Mat.rows, img1Mat.cols, img1Mat.rows));
    img1Mat.copyTo(roi);
    roi.delete();

    // dx, dyにimg1Mat.cols, img1Mat.rowsを加算
    affineMat.doublePtr(0, 2)[0] += img2Mat.cols;
    affineMat.doublePtr(1, 2)[0] += img2Mat.rows;

    // img2Matをアフィン変換して重ねる
    let temp = new cv.Mat.zeros(dsize.height, dsize.width, img2Mat.type());
    cv.warpAffine(img2Mat, temp, affineMat, dsize, cv.INTER_LINEAR, cv.BORDER_TRANSPARENT);

    // マスクで重ねる
    let mask = new cv.Mat();
    cv.cvtColor(temp, mask, cv.COLOR_RGBA2GRAY);
    cv.threshold(mask, mask, 0, 255, cv.THRESH_BINARY);
    temp.copyTo(result, mask);

    // アルファ値が0でない部分のバウンディングボックスを計算してトリミング
    let rgba = new cv.Mat();
    cv.cvtColor(result, rgba, cv.COLOR_RGBA2GRAY);
    let maskNonZero = new cv.Mat();
    cv.threshold(rgba, maskNonZero, 0, 255, cv.THRESH_BINARY);
    let contours = new cv.MatVector();
    let hierarchy = new cv.Mat();
    cv.findContours(maskNonZero, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    if (contours.size() > 0) {
        let rect = cv.boundingRect(contours.get(0));
        for (let i = 1; i < contours.size(); i++) {
            let r = cv.boundingRect(contours.get(i));
            rect.x = Math.min(rect.x, r.x);
            rect.y = Math.min(rect.y, r.y);
            rect.width = Math.max(rect.x + rect.width, r.x + r.width) - rect.x;
            rect.height = Math.max(rect.y + rect.height, r.y + r.height) - rect.y;
        }
        let cropped = result.roi(rect);
        result.delete();
        result = cropped.clone();
        cropped.delete();
    }


    // メモリ解放
    rgba.delete(); maskNonZero.delete(); contours.delete(); hierarchy.delete();
    gray1.delete(); gray2.delete(); akaze.delete(); kp1.delete(); kp2.delete();
    des1.delete(); des2.delete(); bf.delete(); matches.delete();
    affineMat.delete(); temp.delete(); mask.delete();

    return result;
}
