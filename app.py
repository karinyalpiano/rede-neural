# app.py
import os
import numpy as np
import PIL.Image
import tensorflow as tf
import gradio as gr

MODEL_PATH = "mnist_mlp.keras"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Arquivo de modelo não encontrado: {MODEL_PATH}\n"
        "Treine antes com:  py train_mnist.py"
    )
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[OK] Modelo carregado: {MODEL_PATH}")

def predict_from_canvas(canvas_img):
    try:
        if canvas_img is None:
            return {str(i): 0.0 for i in range(10)}

        # Pode vir dict (v5), PIL.Image, ou numpy
        if isinstance(canvas_img, dict):
            for k in ("composite", "image"):
                if k in canvas_img and canvas_img[k] is not None:
                    canvas_img = canvas_img[k]
                    break

        if isinstance(canvas_img, np.ndarray):
            arr = canvas_img
            if arr.dtype != np.uint8:
                arr = arr.astype("uint8")
            if arr.ndim == 2:
                pil = PIL.Image.fromarray(arr, mode="L")
            elif arr.ndim == 3 and arr.shape[2] == 4:
                pil = PIL.Image.fromarray(arr, mode="RGBA").convert("L")
            elif arr.ndim == 3 and arr.shape[2] == 3:
                pil = PIL.Image.fromarray(arr, mode="RGB").convert("L")
            else:
                return {str(i): 0.0 for i in range(10)}
        elif isinstance(canvas_img, PIL.Image.Image):
            pil = canvas_img.convert("L")
        else:
            pil = PIL.Image.fromarray(np.array(canvas_img).astype("uint8")).convert("L")

        pil = pil.resize((28, 28))
        x = np.array(pil).astype("float32") / 255.0
        x = 1.0 - x
        x = x.reshape(1, 28 * 28)

        probs = model.predict(x, verbose=0)[0]
        return {str(i): float(probs[i]) for i in range(10)}
    except Exception as e:
        print("[predict_from_canvas] ERRO:", repr(e))
        return {str(i): 0.0 for i in range(10)}

with gr.Blocks(title="Reconhecedor de Dígitos (MNIST)") as demo:
    gr.Markdown("# ✍️ Reconhecedor de Dígitos (MNIST)\nDesenhe um número de 0 a 9 e clique em **Prever**.")
    with gr.Row():
        canvas = gr.Sketchpad(width=280, height=280, type="numpy")
        output = gr.Label(num_top_classes=10)
    with gr.Row():
        btn_predict = gr.Button("Prever")
        btn_clear = gr.Button("Limpar")

    btn_predict.click(fn=predict_from_canvas, inputs=canvas, outputs=output)
    btn_clear.click(lambda: None, inputs=None, outputs=canvas)

if __name__ == "__main__":
    demo.launch(debug=True)
