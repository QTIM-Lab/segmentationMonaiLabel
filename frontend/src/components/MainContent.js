import React, { useState, useRef } from 'react';

import AddImgArea from './AddImgArea';
import GetImgIdArea from './GetImgIdArea';
import GetInfoArea from './GetInfoArea';
import DisplayImgArea from './DisplayImgArea';
import AddLabelArea from './AddLabelArea';
import ShowInferenceArea from './ShowInferenceArea';
import RunTrainArea from './RunTrainArea';

import SelectEdit from './SelectEdit';

import { Container, Row, Col } from 'react-bootstrap'

// import logo from './logo.svg';
import '../App.css';
import DrawingArea from './DrawingArea';
import DisplayImage from './DisplayImage';
import GetSample from './GetSample';
import PeformInference from './PerformInference';
import SaveLabel from './SaveLabel';
import RunTrain from './RunTrain';

const MainContent = () => {
    const canvasRef = useRef(null);
    const [ctx, setCtx] = useState(null)
    const [imgIdData, setImgIdData] = useState(null);

    const [inferenceResponseData, setInferenceResponseData] = useState(null);
    const [imageResponseData, setImageResponseData] = useState(null);
    const [annotatorResponseData, setAnnotatorResponseData] = useState(null)

  return (
    <div className="App">
      <Container>

          <Row className='my-4'>
              <Col xs={3}>
                <GetSample 
                  setImgIdData={setImgIdData} 
                  imgIdData={imgIdData} 
                  setImageResponseData={setImageResponseData}
                  setInferenceResponseData={setInferenceResponseData}
                  canvasRef={canvasRef}
                  ctx={ctx}
                />
              </Col>

              <Col xs={3}>
                <PeformInference imgId={imgIdData?.id} setInferenceResponseData={setInferenceResponseData} />
              </Col>

              <Col xs={3}>
                <SaveLabel imgId={imgIdData?.id} canvasRef={canvasRef} />
              </Col>

              <Col xs={3}>
                <RunTrain imgId={imgIdData?.id} />
              </Col>
          </Row>

          <Row>
            <Col xs={6}>
              <DisplayImage imgData={imageResponseData} />
            </Col>
            <Col xs={6}>
              <DrawingArea 
                canvasRef={canvasRef}
                ctx={ctx}
                setCtx={setCtx}
                imgId={imgIdData?.id} 
                inferenceResponseData={inferenceResponseData}
                setAnnotatorResponseData={setAnnotatorResponseData}
                imageResponseData={imageResponseData}
              />
            </Col>
          </Row>

          {/* <Row>
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
            

          </Row> */}


      </Container>

        

        

        

        
        
        

        

        

        
        

    </div>
  );
}

export default MainContent;


