import React, { useState } from "react";
import WebcamCapture from "./components/WebcamCapture";
import SignTutorial from "./components/SignTutorial";
import ReactPlayer from "react-player";
import ToggleButton from "./components/ToggleButton";

export default function LearningPage() {

    const [showSignTutorial, setShowSignTutorial] = useState(false);
    const [[currentSign, URL], setCurrentSign] = useState(getNextSign());

    const [showWebcam, setShowWebcam] = useState(true);
    const [isCapturing, setIsCapturing] = React.useState(false);    
    const [shouldAnalyse, setShouldAnalyse] = useState(false);

    const closeSignTutorial = () => setShowSignTutorial(false);
    const displaySignTutorial = () => setShowSignTutorial(true);

    const buttonCSS = "bg-blue-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded";

    /* Should analyse toggle switch state*/

    return (
        <>
            {showSignTutorial && <SignTutorial signName={currentSign!} url={URL!} closeModal={closeSignTutorial} />}
            <div className="p-6 mx-auto bg-slate-200 mt-10 rounded-xl shadow-lg flex flex-col w-fit gap-8">
                <h1 className="text-center text-3xl font-semibold">ASL recognizer: {currentSign === undefined ? "Finished!" : currentSign}</h1>

                <div className="self-center">
                    {/* TODO React player displays "You haven't recorded a video yet" */}
                    {/* TODO React player can play recorded video */}
                    {showWebcam ? 
                        <WebcamCapture isCapturing={isCapturing} setIsCapturing={setIsCapturing} hideWebcam={() => setShowWebcam(false)} shouldAnalyse={shouldAnalyse} signLabel={currentSign!} /> 
                        : 
                        <ReactPlayer url="sign_videos/signvid.webm" controls={true} />}
                </div>
                <div className="self-center flex gap-2">
                    {showWebcam ? 
                        <button onClick={() => setIsCapturing(true)} className={buttonCSS}>Start recording</button>
                        :
                        <button onClick={() => setShowWebcam(true)} className={buttonCSS}>Record</button>
                    }

                    {/* TODO "Check" button, hvis der eksistere en video. Check button laves til "Next" hvis ML siger yes, og ellers laves der et popup til bruger om no fra ML */}
                    {/* ^ Ovenstående er afhængig af om vi vil have auto send (som pt), eller om man skal kunne gennemse og retake */}

                    {showWebcam && <button onClick={() => setShowWebcam(false)} className={buttonCSS}>Video</button>}

                    {currentSign !== undefined && <button onClick={displaySignTutorial} className={buttonCSS}>Tutorial</button>}

                    <button onClick={() => setCurrentSign(getNextSign(currentSign))} className={buttonCSS}>
                        {currentSign === undefined ? "Restart" : "Skip"}
                    </button>

                    <ToggleButton isToggled={shouldAnalyse} setIsToggled={setShouldAnalyse} label="Should analyse" />
                </div>
            </div>
        </>
    );
}

/**
 * @param currentSign the current sign that the user just signed, used to know what the next sign should be
 * @returns [URL, SignName] will return undefined when finished
 */
const getNextSign = (currentSign?: string): [string?, string?] => {

    const signData = require("./models/sign.json");
    const keys = Object.keys(signData);

    if (currentSign === undefined) {
        return [keys[0], signData[keys[0]]];
    }

    for (let i = 0; i <= keys.length; i++) {
        if (currentSign == keys[i])
            return [keys[i+1], signData[keys[i+1]]];
    }
    return [undefined, undefined];
}