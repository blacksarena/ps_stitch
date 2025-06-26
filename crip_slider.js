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
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
            <style>
            .crip-slider-container {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            .crip-slider-label {
                font-size: 0.9em;
                margin-right: 1rem;
            }
            .crip-slider-range {
                flex: 1;
                margin: 0 0.5rem;
            }
            </style>
            <label class="crip-slider-label form-label">${initialLabel}: <span id="valueDisplay" class="fw-bold">${initialValue}</span></label>
            <div class="crip-slider-container mb-2">
            <button id="decrement" class="btn btn-outline-secondary btn-sm" type="button">&lt;</button>
            <input id="slider" class="form-range crip-slider-range" type="range">
            <button id="increment" class="btn btn-outline-secondary btn-sm" type="button">&gt;</button>
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