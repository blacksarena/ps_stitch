class CripSlider extends HTMLElement {
    static get observedAttributes() {
        return ['min', 'max', 'value'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        const initialLabel = this.getAttribute('label') ?? "crip pixel";
        const initialValue = this.getAttribute('value') ?? 50;
        this.shadowRoot.innerHTML = `
            <style>
                .slider-container {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 100%;
                }
                .crip_label {
                    font-size: 0.8em;
                    margin-right: 10px;
                }
                .dec {
                    font-size: 0.5em;
                }
                .crip_slider {
                    width: 100%;
                    margin: 0 10px;
                }
                .inc {
                    margin-left: auto;
                    font-size: 0.5em;
                }
            </style>
            <label class="crip_label">${initialLabel}: <span id="valueDisplay">${initialValue}</span></label>
            <br>
            <div class="slider-container">
                <button id="decrement" class="dec" type="button"><</button>
                <input id="slider" class="crip_slider" type="range">
                <button id="increment" class="inc" type="button">></button>
            </div>
        `;

        this.decrementBtn = this.shadowRoot.getElementById('decrement');
        this.incrementBtn = this.shadowRoot.getElementById('increment');

        this.decrementBtn.addEventListener('click', () => {
            let val = Number(this.slider.value);
            let min = Number(this.slider.min);
            if (val > min) {
                this.slider.value = val - 1;
                this.valueDisplay.textContent = this.slider.value;
                this.setAttribute('value', this.slider.value);
                this.dispatchEvent(new CustomEvent('change', { detail: this.value }));
            }
        });

        this.incrementBtn.addEventListener('click', () => {
            let val = Number(this.slider.value);
            let max = Number(this.slider.max);
            if (val < max) {
                this.slider.value = val + 1;
                this.valueDisplay.textContent = this.slider.value;
                this.setAttribute('value', this.slider.value);
                this.dispatchEvent(new CustomEvent('change', { detail: this.value }));
            }
        });
        this.slider = this.shadowRoot.querySelector('input[type="range"]');
        this.valueDisplay = this.shadowRoot.getElementById('valueDisplay');
        this.slider.addEventListener('input', () => {
            this.valueDisplay.textContent = this.slider.value;
            this.dispatchEvent(new CustomEvent('change', { detail: this.value }));
        });
    }

    connectedCallback() {
        this._updateAttributes();
    }

    attributeChangedCallback(name, oldValue, newValue) {
        this._updateAttributes();
    }

    _updateAttributes() {
        this.slider.min = this.getAttribute('min') ?? 0;
        this.slider.max = this.getAttribute('max') ?? 1000;
        this.slider.value = this.getAttribute('value') ?? 50;
    }

    get value() {
        return this.slider.value;
    }

    set value(val) {
        this.slider.value = val;
        this.setAttribute('value', val);
    }
}

customElements.define('crip-slider', CripSlider);