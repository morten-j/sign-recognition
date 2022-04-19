import React, { useState } from "react";
import VideoPlayer from "./components/VideoPlayer";
import WebcamCapture from "./components/WebcamCapture";
import SignTutorial from "./components/SignTutorial";

export default function LearningPage() {
    const [showWebcam, setShowWebcam] = useState(false)

    const [show, setShow] = useState(false);

    const closeModal = () => setShow(false);
    const showModal = () => setShow(true);

    return (
        <>
            <SignTutorial signName={"Pizza"} show={show} closeModal={closeModal} />
            <div className="p-6 mx-auto bg-white rounded-xl shadow-lg flex flex-col w-2/3">
                <h1 className="text-center">ASL recognizer</h1>
                <div className="self-center">
                    {showWebcam ? <WebcamCapture /> : <VideoPlayer url="sign_videos/signvid.webm" />}
                </div>

                <div className="self-center" >
                    <button onClick={() => setShowWebcam(!showWebcam)} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                        {showWebcam ? "Back to vid" : "Record"}
                    </button>
                </div>
            </div>
        </>
    );
}