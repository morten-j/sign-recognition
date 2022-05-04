import React from "react";

type props = {
    isToggled: boolean,
    setisToggled: React.Dispatch<React.SetStateAction<boolean>>, // react useState setter
    label: string,
}

function ToggleButton({ isToggled, setIsToggled, label }: props) {

    return (
        <div className="flex items-center justify-center">
            <label htmlFor="toogleA" className="flex items-center cursor-pointer">
                <span className="relative">
                    <input id="toogleA" type="checkbox" className="sr-only" checked={isToggled} onChange={() => setIsToggled(!isToggled)}/>
                    <div className="w-10 h-4 bg-gray-400 rounded-full shadow-inner"></div>
                    <div className="dot absolute w-6 h-6 bg-white rounded-full shadow -left-1 -top-1 transition"></div>
                </span>
                <span className="ml-3 text-gray-700 font-medium">
                    {label}
                </span>
            </label>
        </div >
    );
}

export default ToggleButton;