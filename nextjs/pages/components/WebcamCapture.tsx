import Webcam from "react-webcam";
import React from "react";
import Countdown from 'react-countdown';

type Props = {
    isRecording: boolean;
    stopRecording: () => void;
    hideWebcam: () => void;
}

export default function WebcamCapture({ isRecording, stopRecording, hideWebcam} : Props) {
    const webcamRef = React.useRef(null);
    const mediaRecorderRef = React.useRef(null);
    const [capturing, setCapturing] = React.useState(false);
    const [recordedChunks, setRecordedChunks] = React.useState([]);

    const handleStartCaptureClick = React.useCallback(() => {
        setCapturing(true);
        mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
            mimeType: "video/webm"
        });
        mediaRecorderRef.current.addEventListener(
            "dataavailable",
            handleDataAvailable
        );
        mediaRecorderRef.current.start();

    }, [webcamRef, setCapturing, mediaRecorderRef]);

    const handleDataAvailable = React.useCallback(
        ({ data }) => {
            if (data.size > 0) {
                setRecordedChunks((prev) => prev.concat(data));
            }
        },
        [setRecordedChunks]
    );

    const handleStopCaptureClick = React.useCallback(() => {
        mediaRecorderRef.current.stop();
        setCapturing(false);
        
        // mediaRecorderRef.current.requestData();

    }, [mediaRecorderRef, webcamRef, setCapturing]);

    const handleDownload = React.useCallback(() => {
        if (recordedChunks.length) {
            const blob = new Blob(recordedChunks, {
                type: "video/webm"
            });

            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            document.body.appendChild(a);
            a.href = url;
            a.download = "signvid.webm";
            a.click();
            window.URL.revokeObjectURL(url);

            // Send to upload Python server  
            // const fd = new FormData();
            // fd.append("fname", "video.webm")
            // fd.append("video", blob);
            // fetch("http://localhost:8080/api/hands", {
            //     method: "POST",
            //     body: fd,
            // });
        
            setRecordedChunks([]);
        }
    }, [recordedChunks]);

    return (
        <>
            {isRecording && <Countdown date={Date.now() + 3000} onComplete={handleStartCaptureClick} />}
            <Webcam audio={false} ref={webcamRef} mirrored={true} />
            
            {isRecording && recordedChunks.length > 0 && setTimeout(handleStopCaptureClick, 6000) && setTimeout(stopRecording, 6000) && setTimeout(hideWebcam, 8000) && setTimeout(handleDownload, 7000)}
            
        </>
    );
};