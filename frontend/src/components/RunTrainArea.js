import React, { useState } from 'react';

const RunTrainArea = (props) => {

    const [responseData, setResponseData] = useState(null);

    const handleClick = () => {
      // Define the URL and headers
      const url = 'http://0.0.0.0:8000/train/SegformerBundle?run_sync=false&enqueue=false';
      const headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      };
  
      // Define the POST request body data
      const postData = {};
  
      // Send a POST request using the fetch API
      fetch(url, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify(postData),
      })
        .then(response => response.json())
        .then(data => {
          setResponseData(data);
        })
        .catch(error => {
          console.error('Error:', error);
        });
    };
  
    return (
      <div>
        <button onClick={handleClick}>Run Training</button>
        {responseData && (
          <pre>Response: {JSON.stringify(responseData, null, 2)}</pre>
        )}
      </div>
    );
}

export default RunTrainArea;