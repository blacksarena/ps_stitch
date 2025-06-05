// タブの見出し（tab-item）を取得
const tabItems = document.querySelectorAll(".tab-item");

tabItems.forEach((tabItem) => {
    tabItem.addEventListener("click", () => {
        // すべてのタブを非アクティブにする
        tabItems.forEach((t) => {
            t.classList.remove("active");
        });
        // すべてのコンテンツを非表示にする
        const tabPanels = document.querySelectorAll(".tab-panel");
        tabPanels.forEach((tabPanel) => {
            tabPanel.classList.remove("active");
        });

        // クリックされたタブをアクティブにする
        tabItem.classList.add("active");

        // 対応するコンテンツを表示
        const tabIndex = Array.from(tabItems).indexOf(tabItem);
        tabPanels[tabIndex].classList.add("active");
    });
});