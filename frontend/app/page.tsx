"use client";

import { useState } from "react";
import styles from "../styles/Home.module.css";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [epsilon, setEpsilon] = useState<number>(0.1);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [cleanImage, setCleanImage] = useState(null);
const [advImage, setAdvImage] = useState(null);

  const submit = async () => {
    if (!file) {
      alert("Please upload an image first");
      return;
    }

    setLoading(true);
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("epsilon", epsilon.toString());

    try {
      const res = await fetch("https://devneuron-fgsm-assessment-1.onrender.com/attack", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setCleanImage(data.clean_image);
setAdvImage(data.adversarial_image);
      setResult(data);
    } catch (err) {
      console.error(err);
      alert("Failed to connect to backend");
    }
    
    setLoading(false);
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>FGSM Adversarial Attack Demo</h1>
      <p className={styles.subtitle}>
        FastAPI + PyTorch + Next.js
      </p>

      <div className={styles.card}>
        <div className={styles.inputGroup}>
          <label className={styles.label}>Upload Image</label>
          <input
            type="file"
            accept="image/png, image/jpeg"
            className={styles.fileInput}
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
        </div>

        <div className={styles.inputGroup}>
          <label className={styles.label}>
            Epsilon: {epsilon}
          </label>
          <input
            type="range"
            min="0"
            max="0.4"
            step="0.01"
            value={epsilon}
            onChange={(e) => setEpsilon(Number(e.target.value))}
            className={styles.slider}
          />
        </div>

        <button
          className={styles.button}
          onClick={submit}
          disabled={loading}
        >
          {loading ? "Running Attack..." : "Run Attack"}
        </button>

        {result && (
          <div className={styles.results}>
            <p className={styles.resultText}>
              Clean Prediction: <strong>{result.clean_prediction}</strong>
            </p>
            <p className={styles.resultText}>
              Adversarial Prediction:{" "}
              <strong>{result.adversarial_prediction}</strong>
            </p>
            <p
              className={
                result.attack_success
                  ? styles.success
                  : styles.failure
              }
            >
              Attack Success: {result.attack_success.toString()}
            </p>

            <div style={{ display: "flex", gap: "20px", marginTop: "20px" }}>
  <div>
    <h3>Clean Image</h3>
    {cleanImage && (
      <img src={`data:image/png;base64,${cleanImage}`} width="200" />
    )}
  </div>

  <div>
    <h3>Adversarial Image</h3>
    {advImage && (
      <img src={`data:image/png;base64,${advImage}`} width="200" />
    )}
  </div>
</div>
          </div>
        )}
      </div>
    </div>
  );
}
