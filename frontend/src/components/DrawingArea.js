import React, { useState, useEffect } from "react";

const DrawingArea = (props) => {
  
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDrawing, setIsDrawing] = useState(false);
  const [isErasing, setIsErasing] = useState(false);
  const [ctx, setCtx] = useState(null);
  // const [imageData, setImageData] = useState(null);
  const [responseData, setResponseData] = useState(null)

  useEffect(() => {
    if (props.canvasRef.current) {
      setCtx(props.canvasRef.current.getContext("2d"));
    }
  }, [props.canvasRef]);

  // Add a new useEffect to draw the image when props.inferenceResponseData changes.
  useEffect(() => {
    if (props.inferenceResponseData && props.canvasRef.current) {
      const canvas = props.canvasRef.current;
      const image = new Image();
      image.src = props.inferenceResponseData;

      image.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      };
    }
  }, [props.inferenceResponseData, ctx]);

  const handleDown = (event) => {
    setIsDrawing(true);
    const canvasRect = props.canvasRef.current.getBoundingClientRect();
    const x = event.clientX - canvasRect.left;
    const y = event.clientY - canvasRect.top;
    setPosition({ x, y });
  };

  const handleMove = (event) => {
    if (isDrawing) {
      const canvasRect = props.canvasRef.current.getBoundingClientRect();
      const x = event.clientX - canvasRect.left;
      const y = event.clientY - canvasRect.top;

      ctx.beginPath();
      ctx.arc(x, y, 10, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();

      setPosition({ x, y });
    } else if (isErasing) {
      const canvasRect = props.canvasRef.current.getBoundingClientRect();
      const x = event.clientX - canvasRect.left;
      const y = event.clientY - canvasRect.top;

      ctx.clearRect(x - 5, y - 5, 10, 10);
    }
  };

  const handleUp = (event) => {
    if (isDrawing) {
      const canvasRect = props.canvasRef.current.getBoundingClientRect();
      const x = event.clientX - canvasRect.left;
      const y = event.clientY - canvasRect.top;

      ctx.beginPath();
      ctx.moveTo(position.x, position.y);
      ctx.lineTo(x, y);
      ctx.stroke();
    } else if (isErasing) {
      const canvasRect = props.canvasRef.current.getBoundingClientRect();
      const x = event.clientX - canvasRect.left;
      const y = event.clientY - canvasRect.top;

      ctx.clearRect(x - 5, y - 5, 10, 10);
      setIsErasing(false);
    }

    setIsDrawing(false);

    // Grab the image data as base64 bytes.
    // const imageDataBytes = props.canvasRef.current.toDataURL();

    // Update the state variable to store the image data.
    // setImageData(imageDataBytes);
  };

  const uploadImage = () => {
    if (!props.canvasRef.current) {
      alert('Canvas element not available.');
      return;
    }
  
    // Get the canvas element and convert it to a data URL with the "image/png" format.
    const canvas = props.canvasRef.current;
    const imageData = canvas.toDataURL("image/jpeg");
  
    // Check if there's image data to upload.
    if (!imageData) {
      alert('No image data to upload.');
      return;
    }

    console.log("data: ", imageData)
  
    // Define the URL and headers
    const url = `http://0.0.0.0:8000/datastore/label?image=${props.imgId}&tag=website_trying_jpeg_again`;
    const headers = {
      'Accept': 'application/json',
    };
  
    // Create a new FormData object to handle multipart/form-data
    const formData = new FormData();

    formData.append('params', JSON.stringify({}));
  
    // Add the image data as a blob with a "image/png" type
    const formFileName = `${props.imgId}.jpeg`;

    const byteCharacters = atob(imageData.split(',')[1]);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const imageBlob = new Blob([byteArray], { type: 'image/jpeg' });
    formData.append('label', imageBlob, formFileName);
  
    // Send a PUT request with the FormData using the fetch API
    fetch(url, {
      method: 'PUT',
      headers: headers,
      body: formData,
    })
    .then(response => response.json())
    .then(file => {
      setResponseData(file);
    })
    .catch(error => {
      console.error('Error:', error);
    });
  };
  

  const toggleErasing = () => {
    setIsErasing(!isErasing);
  };

  const clearCanvas = () => {
    if (ctx) {
      ctx.clearRect(0, 0, props.canvasRef.current.width, props.canvasRef.current.height);
    }
  };

  const soCanvas = () => {
    console.log("entering")
    debugger
  };


  return (
    <div>
      <canvas
        ref={props.canvasRef}
        width={512}
        height={512}
        onMouseDown={handleDown}
        onMouseMove={handleMove}
        onMouseUp={handleUp}
        style={{ position: "absolute", zIndex: 999, opacity: 0.2 }}
      />
      {/* <img src={props.inferenceResponseData} height={512} width={512} style={{ position: "absolute", opacity: 0.2 }} alt="Pred" /> */}
      <img src={props.imageResponseData} height={512} width={512} alt="Background" />
      
      
      <button onClick={toggleErasing}>
        {isErasing ? "Stop Erasing" : "Erase"}
      </button>
      <button onClick={uploadImage}>Upload Image</button>
      <button onClick={clearCanvas}>Clear Canvas</button>
      <button onClick={soCanvas}>So Canvas</button>
      {responseData && <pre>Response: {JSON.stringify(responseData, null, 2)}</pre>}
    </div>
  );
};

export default DrawingArea;
