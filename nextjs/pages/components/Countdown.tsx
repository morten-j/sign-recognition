import React from "react";

type Props = {
    startSeconds: number;
    startCapture: () => void;
    stopCapture: () => void;
}

const CountDownTimer = ({ startSeconds, startCapture, stopCapture }: Props) => {
    
    const [time, setTime] = React.useState<number>(startSeconds); // Countdown time
    const [started, setStarted] = React.useState<boolean>(false); // Countdown state or capturing state

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