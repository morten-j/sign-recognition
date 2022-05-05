import Webcam from "react-webcam";
import React from "react";
import Countdown from "../components/Countdown";

type Props = {
    isRecording: boolean;
    stopRecording: () => void;
    hideWebcam: () => void;
    shouldAnalyse: boolean;
    signLabel: string;
}

export default function WebcamCapture({ isRecording, stopRecording, hideWebcam, shouldAnalyse, signLabel } : Props) {
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

        stopRecording();
        handleDownload();
        // setTimeout(hideWebcam, 1000);
        hideWebcam();

    }, [mediaRecorderRef, webcamRef, setCapturing]);

    const handleDownload = React.useCallback(() => {
        if (recordedChunks.length) {
            const blob = new Blob(recordedChunks, {
                type: "video/webm"
            });

            // Send to upload Python server  
            const fd = new FormData();
            fd.append("fname", "video.webm")
            fd.append("video", blob);

            // Send to /api/hands if should analyse, else send for video saving only.
            if (shouldAnalyse) {
                fetch("http://localhost:8080/api/hands", {
                    method: "POST",
                    body: fd,
                });
            } else {
                const response = fetch(`http://localhost:8080/api/savevideo?label=${signLabel}`, {
                    method: "POST",
                    body: fd,
                });
                response.then(() => {window.alert("Video saved on server!")})
            }

            setRecordedChunks([]);
        }
    }, [recordedChunks]);

    return (
        <>
            {isRecording && <Countdown startSeconds={3} startCapture={handleStartCaptureClick} stopCapture={handleStopCaptureClick} />}
            <Webcam audio={false} ref={webcamRef} mirrored={true} />
            {capturing ?
                <button onClick={handleStopCaptureClick} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Stop Recording</button>
                :
                <button onClick={handleStartCaptureClick} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Record</button>
            }
            {recordedChunks.length > 0 && <button onClick={handleDownload} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">{shouldAnalyse ? "analyse" : "Save on server"}</button>}
        </>
    );
};