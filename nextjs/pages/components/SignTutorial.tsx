import React from 'react'
import VideoPlayer from './VideoPlayer'

type Props = {
    signName: String,
    show: boolean,
    closeModal: () => void,
}

const SignTutorial = ({signName, show, closeModal}: Props) => {

    const styling = 'w-full h-full bg-gray-900 bg-opacity-80 top-0 fixed sticky-0 z-50'

    return (
    <div className={show ? styling : styling + " hidden"}>
        <div className="2xl:container  2xl:mx-auto py-48 px-4 md:px-28 flex justify-center items-center">
            <div className="mt-12 w-96 md:w-auto dark:bg-gray-800 relative flex flex-col justify-center items-center bg-white py-16 px-4 md:px-24 xl:py-24 xl:px-36 gap-5">

                <button onClick={closeModal} className="text-gray-800 dark:text-gray-400 absolute top-8 right-8 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-800">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M18 6L6 18" stroke="currentColor" stroke-width="1.66667" stroke-linecap="round" stroke-linejoin="round" />
                        <path d="M6 6L18 18" stroke="currentColor" stroke-width="1.66667" stroke-linecap="round" stroke-linejoin="round" />
                    </svg>
                </button>

                <h1 className="text-3xl dark:text-white lg:text-4xl font-semibold leading-7 lg:leading-9 text-center text-gray-800">
                    Make this sign: {signName}
                </h1>

                <VideoPlayer url="signs/signvid.webm" />

                <button onClick={closeModal} className="bg-blue-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
                    I'm Ready!
                </button>

            </div>
        </div>
        
    </div>
    )
}

export default SignTutorial