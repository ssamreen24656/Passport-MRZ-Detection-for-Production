import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [mrzText, setMrzText] = useState("");
  const [mrzFields, setMrzFields] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMrzText("");
    setMrzFields({});
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an image file.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("file", file);

      // Replace with your backend URL if different
      const response = await axios.post("http://localhost:8000/extract_mrz/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setMrzText(response.data.mrz_text);
      setMrzFields(response.data.mrz_fields);
    } catch (err) {
      console.error(err);
      setError("Failed to extract MRZ. Make sure the backend is running.");
    }
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>MRZ Extraction</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button type="submit" disabled={loading}>
          {loading ? "Processing..." : "Upload & Extract MRZ"}
        </button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {mrzText && (
        <div style={{ marginTop: "20px" }}>
          <h2>Raw MRZ Text</h2>
          <pre>{mrzText}</pre>

        </div>
      )}
    </div>
  );
}

export default App;
