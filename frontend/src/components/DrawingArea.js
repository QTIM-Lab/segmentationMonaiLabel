import React, { useState, useEffect } from "react";

import { Row, Button } from "react-bootstrap";
import SelectEdit from "./SelectEdit";

const DrawingArea = (props) => {

  const [selectedOption, setSelectedOption] = useState('Cup');
  
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDrawing, setIsDrawing] = useState(false);
  // const [isErasing, setIsErasing] = useState(false);
  // const [props.ctx, props.setCtx] = useState(null);
  // const [imageData, setImageData] = useState(null);
  const [responseData, setResponseData] = useState(null)
  const [showCanvas, setShowCanvas] = useState(true);

  useEffect(() => {
    if (props.canvasRef.current) {
      props.setCtx(props.canvasRef.current.getContext("2d"));
    }
  }, [props.canvasRef]);

  // Add a new useEffect to draw the image when props.inferenceResponseData changes.
  useEffect(() => {
    if (props.inferenceResponseData && props.canvasRef.current) {
      const canvas = props.canvasRef.current;
      const image = new Image();
      image.src = props.inferenceResponseData;

      image.onload = () => {
        props.ctx.clearRect(0, 0, canvas.width, canvas.height);
        props.ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
      };
    }
  }, [props.inferenceResponseData, props.ctx]);

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

      props.ctx.beginPath();
      props.ctx.arc(x, y, 10, 0, 2 * Math.PI);
      if (selectedOption === "Background") {
        props.ctx.fillStyle = "rgb(0, 0, 255)";
      } else if (selectedOption === "Disc") {
        props.ctx.fillStyle = "rgb(0, 255, 0)";
      } else if (selectedOption === "Cup") {
        props.ctx.fillStyle = "rgb(255, 0, 0)";
      }
      
      props.ctx.fill();

      setPosition({ x, y });
    }
    // } else if (isErasing) {
    //   const canvasRect = props.canvasRef.current.getBoundingClientRect();
    //   const x = event.clientX - canvasRect.left;
    //   const y = event.clientY - canvasRect.top;

    //   props.ctx.clearRect(x - 5, y - 5, 10, 10);
    // }
  };

  const handleUp = (event) => {
    if (isDrawing) {
      const canvasRect = props.canvasRef.current.getBoundingClientRect();
      const x = event.clientX - canvasRect.left;
      const y = event.clientY - canvasRect.top;

      props.ctx.beginPath();
      props.ctx.moveTo(position.x, position.y);
      props.ctx.lineTo(x, y);
      props.ctx.stroke();
    }
    // } else if (isErasing) {
    //   const canvasRect = props.canvasRef.current.getBoundingClientRect();
    //   const x = event.clientX - canvasRect.left;
    //   const y = event.clientY - canvasRect.top;

    //   props.ctx.clearRect(x - 5, y - 5, 10, 10);
    //   setIsErasing(false);
    // }

    setIsDrawing(false);

    // Grab the image data as base64 bytes.
    // const imageDataBytes = props.canvasRef.current.toDataURL();

    // Update the state variable to store the image data.
    // setImageData(imageDataBytes);
  };
  

  // const toggleErasing = () => {
  //   setIsErasing(!isErasing);
  // };

  // const clearCanvas = () => {
  //   if (props.ctx) {
  //     props.ctx.clearRect(0, 0, props.canvasRef.current.width, props.canvasRef.current.height);
  //   }
  // };

  const toggleShowCanvas = () => {
    setShowCanvas(!showCanvas)
  }

  const opacityVal = showCanvas ? 0.15: 0

  return (
    <div>
      <div>
          <canvas
            ref={props.canvasRef}
            width={512}
            height={512}
            onMouseDown={handleDown}
            onMouseMove={handleMove}
            onMouseUp={handleUp}
            style={{ position: "absolute", zIndex: 999, opacity: opacityVal }}
          />
        
        {/* <img src={props.inferenceResponseData} height={512} width={512} style={{ position: "absolute", opacity: 0.2 }} alt="Pred" /> */}
        <img src={props.imageResponseData} height={512} width={512} alt="" />
      </div>
      
      {/* <button onClick={toggleErasing}>
        {isErasing ? "Stop Erasing" : "Erase"}
      </button> */}
      <Row className="my-2">
        <Button onClick={toggleShowCanvas} style={{width: "auto", margin: "auto"}}>
          {
            showCanvas &&
            "Hide "
          }
          {
            !showCanvas &&
            "Show "
          }
          Canvas
        </Button>
        {/* <Button onClick={clearCanvas} style={{width: "auto", margin: "auto"}}>Clear Canvas</Button> */}
      </Row>
      <Row>
        <SelectEdit selectedOption={selectedOption} setSelectedOption={setSelectedOption} />
      </Row>
      
    </div>
  );
};

export default DrawingArea;
