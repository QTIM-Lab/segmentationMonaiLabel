import React, { useState, useRef } from 'react';

import AddImgArea from './AddImgArea';
import GetImgIdArea from './GetImgIdArea';
import GetInfoArea from './GetInfoArea';
import DisplayImgArea from './DisplayImgArea';
import AddLabelArea from './AddLabelArea';
import ShowInferenceArea from './ShowInferenceArea';
import RunTrainArea from './RunTrainArea';

import { Container, Row, Col } from 'react-bootstrap'

// import logo from './logo.svg';
import '../App.css';
import DrawingArea from './DrawingArea';

const MainContent = () => {
    const canvasRef = useRef(null);
    const [imgIdData, setImgIdData] = useState(null);

    const [inferenceResponseData, setInferenceResponseData] = useState(null);
    const [imageResponseData, setImageResponseData] = useState(null);
    const [annotatorResponseData, setAnnotatorResponseData] = useState(null)

  return (
    <div className="App">
      <Container>
          <Row>
            <Col xs={6}>
              <GetInfoArea />
            </Col>

            <Col xs={3}>
              <RunTrainArea />
            </Col>

            <Col xs={3}>
              <AddImgArea />
            </Col>

          </Row>

          <Row>
            <Col xs={2}>
              <GetImgIdArea setData={setImgIdData} imgIdData={imgIdData} />
            </Col>

            <Col xs={5}>
              <DisplayImgArea imgId={imgIdData?.id} imageResponseData={imageResponseData} setImageResponseData={setImageResponseData} setInferenceResponseData={setInferenceResponseData} />
            </Col>

            <Col xs={5}>
              <ShowInferenceArea 
                imgId={imgIdData?.id} 
                inferenceResponseData={inferenceResponseData} 
                setInferenceResponseData={setInferenceResponseData} 
              />
            </Col>


          </Row>

          <Row>
            <Col xs={2}>
            
            </Col>

            <Col xs={5}>
              <DrawingArea 
                canvasRef={canvasRef}
                imgId={imgIdData?.id} 
                inferenceResponseData={inferenceResponseData}
                setAnnotatorResponseData={setAnnotatorResponseData}
                imageResponseData={imageResponseData}
              />
            </Col>

            <Col xs={5}>
              <AddLabelArea imgId={imgIdData?.id} canvasRef={canvasRef} />
            </Col>
            

          </Row>


      </Container>

        

        

        

        
        
        

        

        

        
        

    </div>
  );
}

export default MainContent;


