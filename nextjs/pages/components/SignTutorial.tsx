import React, { useEffect, useState } from 'react'
import ReactPlayer from 'react-player'
import useKeyPress from './hooks/useKeyPress'

type Props = {
    signName: String,
    show: boolean,
    closeModal: () => void,
}

const SignTutorial = ({signName, show, closeModal}: Props) => {

    const styling = 'w-full h-full bg-gray-900 bg-opacity-80 top-0 fixed sticky-10s z-50'

    const escapeKeyListener = useKeyPress("Escape", closeModal);

    return (
    <div onClick={closeModal} className={show ? styling : styling + " hidden"}>
            <div onClick={(e) => e.stopPropagation()} className="w-fit px-20 m-auto flex justify-center items-center mt-12 relative flex flex-col justify-center items-center bg-white py-8 gap-5 rounded">

                <button onClick={closeModal} className="text-gray-800 absolute top-8 right-8 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-800">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 6L6 18" stroke="currentColor" stroke-width="1.66667" stroke-linecap="round" stroke-linejoin="round" />
                        <path d="M6 6L18 18" stroke="currentColor" stroke-width="1.66667" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                </button>

                <h1 className="text-3xl font-semibold leading-7 text-center">
                    Make this sign: {signName}
                </h1>

                <ReactPlayer url="signs/signvid.webm" controls={false} />

                <button onClick={closeModal} className="bg-blue-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
                    I'm Ready!
                </button>

            </div>
    </div>
    )
}

export default SignTutorial