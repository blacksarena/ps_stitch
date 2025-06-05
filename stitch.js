
// sa https://qiita.com/ka10ryu1/items/bd05aed321a7a154d8a1
let imgMats = [];

document.getElementById("load").addEventListener("click", () => {
    document.getElementById("imgs").click();
});

document.getElementById('imgs').onchange = function (e) {
    imgMats.forEach(mat => mat && mat.delete && mat.delete());
    imgMats = [];
    let files = Array.from(e.target.files);
    files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
    let loaded = 0;
    files.forEach((file, idx) => {
        let img = new Image();
        img.onload = function () {
            let canvas = document.createElement('canvas');
            canvas.width = img.width; canvas.height = img.height;
            canvas.getContext('2d').drawImage(img, 0, 0);
            imgMats[idx] = cv.imread(canvas);
            loaded++;
        };
        console.log(`Loading image ${idx + 1}/${files.length}: ${file.name}`);
        img.src = URL.createObjectURL(file);
    });
};

const cropHRange = document.getElementById('cropHRange');
const cropHValue = document.getElementById('cropHValue');
if (cropHRange && cropHValue) {
    cropHRange.addEventListener('input', function () {
        cropHValue.textContent = cropHRange.value + '%';
    });
}
const cropVRange = document.getElementById('cropVRange');
const cropVValue = document.getElementById('cropVValue');
if (cropVRange && cropVValue) {
    cropVRange.addEventListener('input', function () {
        cropVValue.textContent = cropVRange.value + '%';
    });
}

function stitch() {
    if (imgMats.length < 2) {
        alert("2枚以上の画像を選択してください");
        return;
    }
    let base = imgMats[0].clone();
    for (let i = 1; i < imgMats.length; i++) {
        // let result = stitchPairAffine(base, imgMats[i]);
        let result = stitchPairAffineAkaze(base, imgMats[i]);
        base.delete();
        base = result;
    }
    cv.imshow('stitched_image', base);
    base.delete();
    document.getElementById('download').disabled = false;
}

// 2枚の画像を平面前提でstitchする
function stitchPair(img1Mat, img2Mat) {
    let gray1 = new cv.Mat();
    let gray2 = new cv.Mat();
    cv.cvtColor(img1Mat, gray1, cv.COLOR_RGB2GRAY);
    cv.cvtColor(img2Mat, gray2, cv.COLOR_RGB2GRAY);

    let orb = new cv.ORB();
    let kp1 = new cv.KeyPointVector();
    let des1 = new cv.Mat();
    orb.detectAndCompute(gray1, new cv.Mat(), kp1, des1);

    let kp2 = new cv.KeyPointVector();
    let des2 = new cv.Mat();
    orb.detectAndCompute(gray2, new cv.Mat(), kp2, des2);

    let bf = new cv.BFMatcher(cv.NORM_HAMMING, true);
    let matches = new cv.DMatchVector();
    bf.match(des1, des2, matches);

    // マッチング結果を距離でソートし、上位のものだけを保持
    let matchesArr = [];
    for (let i = 0; i < matches.size(); i++) {
        matchesArr.push(matches.get(i));
    }
    matchesArr.sort((a, b) => a.distance - b.distance);
    matches = new cv.DMatchVector();
    for (let m of matchesArr) {
        matches.push_back(m);
    }
    // --- ここを入力値で調整 ---
    let keep = parseInt(document.getElementById('matchNum').value, 10);
    let filtered = new cv.DMatchVector();
    for (let i = 0; i < Math.min(keep, matches.size()); i++) {
        filtered.push_back(matches.get(i));
    }
    matches.delete();
    matches = filtered;

    let srcPoints = [];
    let dstPoints = [];
    for (let i = 0; i < matches.size(); i++) {
        let m = matches.get(i);
        srcPoints.push(kp1.get(m.queryIdx).pt.x, kp1.get(m.queryIdx).pt.y);
        dstPoints.push(kp2.get(m.trainIdx).pt.x, kp2.get(m.trainIdx).pt.y);
    }
    let srcMat = cv.matFromArray(matches.size(), 1, cv.CV_32FC2, srcPoints);
    let dstMat = cv.matFromArray(matches.size(), 1, cv.CV_32FC2, dstPoints);

    let H = cv.findHomography(dstMat, srcMat, cv.RANSAC);

    // 平面前提なので、横に広げる
    let dsize = new cv.Size(img1Mat.cols + img2Mat.cols, Math.max(img1Mat.rows, img2Mat.rows));
    let result = new cv.Mat.zeros(dsize.height, dsize.width, img1Mat.type());

    // まずimg1Matを左側に貼る
    let roi = result.roi(new cv.Rect(0, 0, img1Mat.cols, img1Mat.rows));
    img1Mat.copyTo(roi);
    roi.delete();

    // img2Matをワープして右側に重ねる
    let temp = new cv.Mat();
    cv.warpPerspective(img2Mat, temp, H, dsize);
    // 透明部分を考慮して加算
    let mask = new cv.Mat();
    cv.cvtColor(temp, mask, cv.COLOR_RGBA2GRAY);
    cv.threshold(mask, mask, 0, 255, cv.THRESH_BINARY);
    temp.copyTo(result, mask);

    // メモリ解放
    gray1.delete(); gray2.delete();
    orb.delete(); kp1.delete(); kp2.delete(); des1.delete(); des2.delete();
    bf.delete(); matches.delete();
    srcMat.delete(); dstMat.delete(); H.delete();
    temp.delete(); mask.delete();

    return result;
}
function stitchPairAffine(img1Mat, img2Mat) {
    // グレースケール変換
    let gray1 = new cv.Mat();
    let gray2 = new cv.Mat();
    cv.cvtColor(img1Mat, gray1, cv.COLOR_RGB2GRAY);
    cv.cvtColor(img2Mat, gray2, cv.COLOR_RGB2GRAY);

    // ORB特徴点抽出
    let orb = new cv.ORB();
    let kp1 = new cv.KeyPointVector();
    let des1 = new cv.Mat();
    orb.detectAndCompute(gray1, new cv.Mat(), kp1, des1);

    let kp2 = new cv.KeyPointVector();
    let des2 = new cv.Mat();
    orb.detectAndCompute(gray2, new cv.Mat(), kp2, des2);

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
        gray1.delete(); gray2.delete(); orb.delete(); kp1.delete(); kp2.delete();
        des1.delete(); des2.delete(); bf.delete(); matches.delete();
        return img1Mat.clone();
    }

    // 2組のマッチ点
    let m0 = matchesArr[0];
    let m1 = matchesArr[1];

    let p1_0 = kp1.get(m0.queryIdx).pt;
    let p2_0 = kp2.get(m0.trainIdx).pt;
    let p1_1 = kp1.get(m1.queryIdx).pt;
    let p2_1 = kp2.get(m1.trainIdx).pt;

    // スケール計算
    let d1 = Math.hypot(p1_1.x - p1_0.x, p1_1.y - p1_0.y);
    let d2 = Math.hypot(p2_1.x - p2_0.x, p2_1.y - p2_0.y);
    let scale = d1 > 0 && d2 > 0 ? d1 / d2 : 1.0;

    // 平行移動計算（スケール適用後）
    let dx = p1_0.x - p2_0.x * scale;
    let dy = p1_0.y - p2_0.y * scale;

    // アフィン行列（回転なし、スケール＋平行移動のみ）
    let affineMat = cv.matFromArray(2, 3, cv.CV_64F, [
        scale, 0, dx,
        0, scale, dy
    ]);

    // 合成用画像
    let dsize = new cv.Size(img1Mat.cols + img2Mat.cols, Math.max(img1Mat.rows, img2Mat.rows));
    let result = new cv.Mat.zeros(dsize.height, dsize.width, img1Mat.type());

    // img1Matを左側に貼る
    let roi = result.roi(new cv.Rect(0, 0, img1Mat.cols, img1Mat.rows));
    img1Mat.copyTo(roi);
    roi.delete();

    // img2Matをアフィン変換して重ねる
    let temp = new cv.Mat();
    cv.warpAffine(img2Mat, temp, affineMat, dsize, cv.INTER_LINEAR, cv.BORDER_TRANSPARENT);

    // マスクで重ねる
    let mask = new cv.Mat();
    cv.cvtColor(temp, mask, cv.COLOR_RGBA2GRAY);
    cv.threshold(mask, mask, 0, 255, cv.THRESH_BINARY);
    temp.copyTo(result, mask);

    // メモリ解放
    gray1.delete(); gray2.delete(); orb.delete(); kp1.delete(); kp2.delete();
    des1.delete(); des2.delete(); bf.delete(); matches.delete();
    affineMat.delete(); temp.delete(); mask.delete();

    return result;
}
function stitchPairAffineAkaze(img1Mat, img2Mat) {
    function cropCenter(mat, cropRatioW = 0.7, cropRatioH = 0.7) {
        let w = mat.cols, h = mat.rows;
        let cropW = Math.floor(w * cropRatioW);
        let cropH = Math.floor(h * cropRatioH);
        let x = Math.floor((w - cropW) / 2);
        let y = Math.floor((h - cropH) / 2);
        let rect = new cv.Rect(x, y, cropW, cropH);
        let cropped = mat.roi(rect);
        return { cropped, offset: { x, y } };
    }

    let crop1 = cropCenter(img1Mat, cropHRange.value / 100, cropVRange.value / 100);
    let crop2 = cropCenter(img2Mat, cropHRange.value / 100, cropVRange.value / 100);
    let img1Cropped = crop1.cropped;
    let img2Cropped = crop2.cropped;
    let offset1 = crop1.offset;
    let offset2 = crop2.offset;

    // グレースケール変換
    let gray1 = new cv.Mat();
    let gray2 = new cv.Mat();
    cv.cvtColor(img1Cropped, gray1, cv.COLOR_RGB2GRAY);
    cv.cvtColor(img2Cropped, gray2, cv.COLOR_RGB2GRAY);

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

    // 2組のマッチ点
    let m0 = matchesArr[0];
    let m1 = matchesArr[1];

    let p1_0 = kp1.get(m0.queryIdx).pt;
    let p2_0 = kp2.get(m0.trainIdx).pt;
    let p1_1 = kp1.get(m1.queryIdx).pt;
    let p2_1 = kp2.get(m1.trainIdx).pt;

    // 元画像の座標系に戻す
    let p1_0_org = { x: p1_0.x + offset1.x, y: p1_0.y + offset1.y };
    let p2_0_org = { x: p2_0.x + offset2.x, y: p2_0.y + offset2.y };
    let p1_1_org = { x: p1_1.x + offset1.x, y: p1_1.y + offset1.y };
    let p2_1_org = { x: p2_1.x + offset2.x, y: p2_1.y + offset2.y };

    // スケール計算
    let d1 = Math.hypot(p1_1_org.x - p1_0_org.x, p1_1_org.y - p1_0_org.y);
    let d2 = Math.hypot(p2_1_org.x - p2_0_org.x, p2_1_org.y - p2_0_org.y);
    let scale = d1 > 0 && d2 > 0 ? d1 / d2 : 1.0;

    // 平行移動計算（スケール適用後）
    let dx = p1_0_org.x - p2_0_org.x * scale;
    let dy = p1_0_org.y - p2_0_org.y * scale;

    // アフィン行列（回転なし、スケール＋平行移動のみ）
    let affineMat = cv.matFromArray(2, 3, cv.CV_64F, [
        scale, 0, dx,
        0, scale, dy
    ]);

    // 合成用画像
    let dsize = new cv.Size(img1Mat.cols + img2Mat.cols, Math.max(img1Mat.rows, img2Mat.rows));
    let result = new cv.Mat.zeros(dsize.height, dsize.width, img1Mat.type());

    // img1Matを左側に貼る
    let roi = result.roi(new cv.Rect(0, 0, img1Mat.cols, img1Mat.rows));
    img1Mat.copyTo(roi);
    roi.delete();

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