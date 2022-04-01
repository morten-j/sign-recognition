// Next.js API route support: https://nextjs.org/docs/api-routes/introduction
import type { NextApiRequest, NextApiResponse } from 'next'

export default function uploadVideo(req: NextApiRequest, res: NextApiResponse) {
    try {
        fetch("<url>", {method: "POST", body: req.body}).then(()=>{
            res.status(200).json({"Status": "Success"});
        });
    } catch (e) {
        res.status(500).json({"Status": "Failed"});
    }
}