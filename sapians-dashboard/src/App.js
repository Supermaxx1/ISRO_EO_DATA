import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [query, setQuery] = useState('');
  const [result, setResult] = useState(null);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('image', image);
    formData.append('query', query);
    try {
      const response = await axios.post('http://localhost:8000/analyze', formData);
      setResult(response.data);
    } catch (err) {
      setResult({ status: 'error', error: err.message });
    }
  };

  return (
    <div style={{ maxWidth: "480px", margin: "auto", padding: "32px" }}>
      <h1>ISRO EO Image Analyzer</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".tif,.png,.jpg" onChange={handleImageChange} required />
        <input type="text" value={query} onChange={e => setQuery(e.target.value)} placeholder="Enter your query" required />
        <button type="submit">Analyze</button>
      </form>
      {result && (
        <div style={{ marginTop: "32px" }}>
          <h3>Results</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
