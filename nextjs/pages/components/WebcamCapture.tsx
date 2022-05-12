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
                fetch("http://localhost:8000/api/hands", 
                    {
                        method: "POST",
                        body: fd,
                    }
                );
            } else {
                const response = fetch(`http://localhost:8000/api/savevideo?label=${signLabel}`, 
                    {
                        method: "POST",
                        body: fd,
                    }
                );
                response.then(() => {
                    window.alert("Video saved on server!"); 
                    hideWebcam();
                })
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