class CripSlider extends HTMLElement {
    static get observedAttributes() {
        return ['min', 'max', 'value'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        const initialLabel = this.getAttribute('label') ?? 'Slider';
        const initialValue = this.getAttribute('value') ?? 0;

        let slider_value = getCookieValue(initialLabel);
        if (slider_value !== null) {
            slider_value = Math.max(Number(slider_value), Number(initialValue));
        } else {
            slider_value = initialValue;
        }

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
            <label class="crip-slider-label form-label">${initialLabel}: <span id="valueDisplay" class="fw-bold">${slider_value}</span></label>
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
                this.dispatchEvent(new CustomEvent('change', { detail: this.value }));
                setCookie(initialLabel, this.slider.value);
                updatePreview();
            }
        });

        this.incrementBtn.addEventListener('click', () => {
            let val = Number(this.slider.value);
            let max = Number(this.slider.max);
            if (val < max) {
                this.slider.value = val + 1;
                this.valueDisplay.textContent = this.slider.value;
                this.dispatchEvent(new CustomEvent('change', { detail: this.value }));
                setCookie(initialLabel, this.slider.value);
                updatePreview();
            }
        });
        this.slider = this.shadowRoot.querySelector('input[type="range"]');
        this.valueDisplay = this.shadowRoot.getElementById('valueDisplay');
        this.slider.addEventListener('input', () => {
            this.valueDisplay.textContent = this.slider.value;
            this.dispatchEvent(new CustomEvent('change', { detail: this.value }));
            setCookie(initialLabel, this.slider.value);
            updatePreview();
        });
        this.slider.min = this.getAttribute('min') ?? 0;
        const labelLower = initialLabel.toLowerCase();
        if (labelLower.includes('top') || labelLower.includes('bottom')) {
            this.slider.max = window.innerHeight;
        } else if (labelLower.includes('right') || labelLower.includes('left')) {
            this.slider.max = window.innerWidth;
        }
        this.slider.value = slider_value;
        this.valueDisplay.textContent = this.slider.value;
    }

    get value() {
        return this.slider.value;
    }

    set value(val) {
        this.slider.value = val;
    }
}

customElements.define('crip-slider', CripSlider);