import React from "react";

interface ICountdown {
    startSeconds: number;
    startCapture: () => void;
    stopCapture: () => void;
}

const CountDownTimer = ({ startSeconds, startCapture, stopCapture }: ICountdown) => {
    
    const [time, setTime] = React.useState<number>(startSeconds); // Countdown time
    const [started, setStart] = React.useState<boolean>(false); // Countdown state or capturing state

    const tick = () => {
        
        // equal 1, to avoid 0 in countdown text
        if (time === 1)  {
            if (!started) {
                setTime(startSeconds); // reset timer
                startCapture();
                setStarted(true);
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