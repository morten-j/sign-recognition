import React from "react";
import VideoPlayer from "./components/VideoPlayer";
import WebcamCapture from "./components/WebcamCapture";

export default function LearningPage() {
  const [showWebcam, setShowWebcam] = React.useState(false)

  return (
    <>
      <div className="p-6 mx-auto bg-white rounded-xl shadow-lg flex flex-col w-2/3">
        <h1 className="text-center">ASL recognizer</h1>
        <div className="self-center">
          { showWebcam ? null : <VideoPlayer /> }
          { showWebcam ? <WebcamCapture /> : null }
        </div>
      
        <div className="self-center" >
          {/* buttons */}
          { showWebcam ? null : <button onClick={() => setShowWebcam(true)} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Record</button> }
          { showWebcam ? <button onClick={() => setShowWebcam(false)} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">Back to vid</button> : null }
        </div>
      </div>
    </>
  );
}