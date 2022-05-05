import React from "react";
import { start } from "repl";

interface ICountdown {
    startSeconds : number;
    startCapture : () => void;
    stopCapture : () => void;
}

const CountDownTimer = ({ startSeconds, startCapture, stopCapture }: ICountdown) => {
    
    const [time, setTime] = React.useState<number>(startSeconds);
    const [started, setStart] = React.useState<boolean>(false);

    const tick = () => {
   
        if (time === 1)  {
            if (!started) {
                setTime(startSeconds);
                startCapture();
                setStart(true);
            }
            else 
                stopCapture();
        } else
            setTime(time - 1);
    };

    React.useEffect(() => {
        const timerId = setInterval(() => tick(), 1000);
        return () => clearInterval(timerId);
    });

    return (
        <p>{`${time} ${started ? ": Started recording" : ""}`}</p> 
    );
}

export default CountDownTimer;