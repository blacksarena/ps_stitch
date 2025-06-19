
function downloadCanvas() {
    const canvas = document.getElementById('stitched_image');
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    const now = new Date();
    const pad = n => n.toString().padStart(2, '0');
    const fileName = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}.png`;
    link.download = fileName;
    link.click();
}