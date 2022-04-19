import React, { useState } from "react";
import VideoPlayer from "./components/VideoPlayer";
import WebcamCapture from "./components/WebcamCapture";
import SignTutorial from "./components/SignTutorial";

export default function LearningPage() {
    const [showWebcam, setShowWebcam] = useState(false)

    const [showSignTutorial, setShowSignTutorial] = useState(false);

    const closeSignTutorial = () => setShowSignTutorial(false);
    const displaySignTutorial = () => setShowSignTutorial(true);

    const buttonCSS = "bg-blue-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded";

    return (
        <>
            <SignTutorial signName={"Pizza"} show={showSignTutorial} closeModal={closeSignTutorial} />
            <div className="p-6 mx-auto bg-slate-200 my-10 rounded-xl shadow-lg flex flex-col w-2/3 gap-8">
                <h1 className="text-center text-3xl font-semibold">ASL recognizer</h1>
                <div className="self-center">
                    {showWebcam ? <WebcamCapture /> : <VideoPlayer url="sign_videos/signvid.webm" />}
                </div>

                <div className="self-center flex gap-2">
                    <button onClick={() => setShowWebcam(!showWebcam)} className={buttonCSS}>
                        {showWebcam ? "Back to vid" : "Record"}
                    </button>
                    <button onClick={displaySignTutorial} className={buttonCSS}>Show Sign Tutorial</button>
                </div>
            </div>
        </>
    );
}