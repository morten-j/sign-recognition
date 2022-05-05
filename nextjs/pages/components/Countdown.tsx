import React from "react";

interface ICountdown {
    hours : number;
    minutes : number;
    seconds : number;
}

interface ICountdownInput extends ICountdown{
    hours : number;
    minutes : number;
    seconds : number;
    callback : () => void;
}

const CountDownTimer = ({ hours = 0, minutes = 0, seconds, callback }: ICountdownInput) => {

    const [time, setTime] = React.useState<ICountdown>({hours, minutes, seconds});

    const tick = () => {

        if (time.hours === 0 && time.minutes === 0 && time.seconds === 0)
            reset()
        else if (time.hours === 0 && time.seconds === 0) {
            setTime({hours: time.hours - 1, minutes: 59, seconds: 59});
        } else if (time.seconds === 0) {
            setTime({hours: time.hours, minutes: time.minutes - 1, seconds: 59});
        } else {
            setTime({hours: time.hours, minutes: time.minutes, seconds: time.seconds - 1});
        }
    };

    const reset = () => setTime({hours: time.hours, minutes: time.minutes, seconds: time.seconds});

    React.useEffect(() => {
        const timerId = setInterval(() => tick(), 1000);
        return () => clearInterval(timerId);
    });

    return (
        <div>
            <p>{`${time.seconds.toString()}`}</p>
        </div>
    );
}

export default CountDownTimer;