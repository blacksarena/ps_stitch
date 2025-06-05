
function downloadCanvas() {
    const canvas = document.getElementById('stitched_image');
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'result.png';
    link.click();
}