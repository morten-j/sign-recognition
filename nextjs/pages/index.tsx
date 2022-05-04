import React, { useState } from "react";
import WebcamCapture from "./components/WebcamCapture";
import SignTutorial from "./components/SignTutorial";
import ReactPlayer from "react-player";
import ToggleButton from "./components/ToggleButton";

export default function LearningPage() {

    const buttonCSS = "bg-blue-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded";

    const [showWebcam, setShowWebcam] = useState(false)
    const [showSignTutorial, setShowSignTutorial] = useState(false);

    const closeSignTutorial = () => setShowSignTutorial(false);
    const displaySignTutorial = () => setShowSignTutorial(true);

    const [[currentSign, URL], setCurrentSign] = useState(getNextSign());

    /* Should analyse toggle switch state*/
    const [shouldAnalyse, setShouldAnalyse] = useState(false);

    return (
        <>
            {showSignTutorial && <SignTutorial signName={currentSign!} url={URL!} closeModal={closeSignTutorial} />}

            <div className="p-6 mx-auto bg-slate-200 mt-10 rounded-xl shadow-lg flex flex-col w-fit gap-8">
                <h1 className="text-center text-3xl font-semibold">ASL recognizer: {currentSign === undefined ? "Finished!" : currentSign}</h1>

                <div className="self-center">
                    {showWebcam ? <WebcamCapture shouldAnalyse={shouldAnalyse} signLabel={currentSign!} /> : <ReactPlayer url="sign_videos/signvid.webm" controls={true} />}
                </div>

                <div className="self-center flex gap-2">
                    <button onClick={() => setCurrentSign(getNextSign(currentSign))} className={buttonCSS}>
                        {currentSign === undefined ? "Restart" : "Next Sign"}
                    </button>

                    <button onClick={() => setShowWebcam(!showWebcam)} className={buttonCSS}>
                        {showWebcam ? "Back to vid" : "Record"}
                    </button>

                    {currentSign !== undefined && <button onClick={displaySignTutorial} className={buttonCSS}>Show Sign Tutorial</button>}

                    <ToggleButton isToggled={shouldAnalyse} setisToggled={setShouldAnalyse} label="should analyse" />
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