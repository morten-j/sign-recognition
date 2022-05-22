import Webcam from "react-webcam";
import React from "react";
import Countdown from "../components/Countdown";

type Props = {
    isCapturing : boolean;
    setIsCapturing: React.Dispatch<React.SetStateAction<boolean>>;
    hideWebcam: () => void;
    shouldAnalyse: boolean;
    signLabel: string;
    setBlobURL: React.Dispatch<React.SetStateAction<string>>;
}

// Response object from Sanic
type ResponseJSON = {
    prediction: string;
    allPredictions: { sign: string, certainty: number }[];
}

export default function WebcamCapture({ isCapturing, setIsCapturing, hideWebcam, shouldAnalyse, signLabel, setBlobURL } : Props) {
    const webcamRef = React.useRef(null);
    const mediaRecorderRef = React.useRef(null);
    const [recordedChunks, setRecordedChunks] = React.useState([]);

    const handleStartCaptureClick = React.useCallback(() => {
        setIsCapturing(true);
        mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
            mimeType: "video/webm"
        });
        mediaRecorderRef.current.addEventListener(
            "dataavailable",
            handleDataAvailable
        );
        mediaRecorderRef.current.start();

    }, [webcamRef, setIsCapturing, mediaRecorderRef]);

    const handleDataAvailable = React.useCallback(
        ({ data }) => {
            if (data.size > 0) {
                setRecordedChunks((prev) => prev.concat(data));
            }
        }, [setRecordedChunks]
    );

    const handleStopCaptureClick = React.useCallback(() => {
        mediaRecorderRef.current.stop();
        setIsCapturing(false);

    }, [mediaRecorderRef, webcamRef, setIsCapturing]);

    const handleDownload = React.useCallback(() => {
        if (recordedChunks.length) {
            const blob = new Blob(recordedChunks, {
                type: "video/webm"
            });

            setBlobURL(URL.createObjectURL( blob ));

            // Send to upload Python server  
            const fd = new FormData();
            fd.append("fname", "video.webm")
            fd.append("video", blob);

            // Send to /api/hands if should analyse, else send for video saving only.
            if (shouldAnalyse) {

                fetch("http://localhost:8080/api/predict", 
                    {
                        method: "POST",
                        body: fd,
                    }
                ).then((response) => {

                    response.json().then((jsonData) => {

                   
                        const predictObject: ResponseJSON = JSON.parse(jsonData);
                        
                        if (predictObject.prediction == signLabel) {
                            window.alert("You did the sign correctly!")
                        } else {

                            let alertString: string = `Incorrect, recognised sign ${predictObject.prediction} and not ${signLabel}\n`;
                            const predictions = predictObject.allPredictions.sort((a, b) => (a.certainty > b.certainty ? -1 : 1));

                            // Build multiline alert string with. Skip first because it is .prediction
                            for (let i=1; i < predictions.length; i++)
                                alertString += `${predictions[i].sign}: ${predictions[i].certainty*100}%\n`;

                            window.alert(alertString);
                        }
                    });
                });
                hideWebcam();
            } else {
                
                await fetch(`http://localhost:8080/api/savevideo?label=${signLabel}`, 
                    {
                        method: "POST",
                        body: fd,
                    }
                )
                window.alert("Video saved on server!"); 
                hideWebcam();
            }
            setRecordedChunks([]);
        }
    }, [recordedChunks]);

    return (
        <>
            {isCapturing && <Countdown startSeconds={3} startCapture={handleStartCaptureClick} stopCapture={handleStopCaptureClick} />}
            <Webcam audio={false} ref={webcamRef} mirrored={true} />

            {/* Once recordedChunks is availble execute func */}
            {recordedChunks.length > 0 && handleDownload()}
        </>
    );
};