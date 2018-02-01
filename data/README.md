# Dataset on YouTube engagement study

## Sofware
  An integrated YouTube data crawler - [YouTube insight data crawler](https://github.com/avalanchesiqi/youtube-insight).

## Data
  The data is hosted on [Google drive](https://drive.google.com/drive/folders/1xCEJU5P185tDJIk-XNV3cckfFAW3W1Px?usp=sharing).
  
  ```
  data
  │   README.md
  └───tweeted_videos.tar.bz2
  │   │   autos.json
  │   │   comedy.json
  │   │   education.json
  │   │   entertainment.json
  │   │   film.json
  │   │   gaming.json
  │   │   howto.json
  │   │   movies.json
  │   │   music.json
  │   │   news.json
  │   │   nonprofits.json
  │   │   people.json
  │   │   pets.json
  │   │   science.json
  │   │   shows.json
  │   │   sports.json
  |   |   trailers.json
  │   │   travel.json
  └───quality_videos.tar.bz2
      │   billboard16.json
      |   top_news.json
      │   vevo.json
  ```

### File Description
All files are in `json` format and compressed in `tar.bz2`.

Dataset | Uncompressed | Ccompressed | #Videos | #Channels
--- | --- | --- | --- | ---
Tweeted videos | 26GB | 4.6GB | 5,331,204 | 1,257,412
Quality videos | 1.9GB | 359MB | 96,397 | 8,823
VEVO videos | 1.4GB | - | 67,649 | 8,685
Billboard16 videos | 1.1MB | - | 63 | 47
Top news videos | 469MB | - | 28,685 | 91

## Data Fields
Each line is a YouTube video, an example is shown below.
```json
{
   "id": "pFMj8KL8nJA",
   "snippet": {
      "description": "For more on India's goods and services tax and the future of the economy under Prime Minister Narendra Modi, CCTV America\u2019s Rachelle Akuffo interviewed Peter Kohli, the chief investment officer at D-M-S Funds.",
      "title": "Peter Kohli on the importance of the goods and services tax",
      "channelId": "UCj7wKsOBhRD9Jy4yahkMRMw",
      "channelTitle": "CCTV America",
      "publishedAt": "2016-08-10T00:34:01.000Z",
      "categoryId": "25",
      "detectLang": "en"
   },
   "contentDetails": {
      "duration": "PT5M27S",
      "definition": "hd",
      "dimension": "2d",
      "caption": "false"
   },
   "topicDetails": {
      "topicIds": ["/m/0546cd"],
      "relevantTopicIds": ["/m/03rk0", "/m/0gfps3", "/m/0296q2", "/m/05qt0", "/m/0dgrhmk", "/m/09x0r", "/m/05qt0", "/m/098wr"]
   },
   "insights": {
      "startDate": "2016-08-10",
      "days": "0,1,2,3,4,5,6,7,8,10,11,14,15,16,17,18,19,23,26,29,30,44,45,62,69,114,118,122,149,154,159,160,182,188,189,199,204,226,253",
      "dailyView": "70,11,15,7,7,8,11,4,7,2,2,1,6,6,3,2,2,2,1,1,4,1,1,1,1,2,3,1,1,1,1,3,1,2,2,1,1,1,1",
      "totalView": "281",
      "dailyWatch": "171.966666667,22.35,42.95,24.6333333333,26.05,25.3833333333,34.25,9.63333333333,6.31666666667,0.7,7.13333333333,0.0333333333333,15.2333333333,16.7,2.2,0.116666666667,0.966666666667,1.1,5.43333333333,5.43333333333,10.7666666667,1.2,5.43333333333,1.8,5.43333333333,5.45,3.15,0.2,1.68333333333,0.733333333333,0.483333333333,3.21666666667,5.43333333333,0.383333333333,5.6,0.0666666666667,0.533333333333,5.43333333333,1.06666666667",
      "avgWatch": "2.3290628707",
      "dailyShare": "2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
      "totalShare": "2",
      "dailySubscriber": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
      "totalSubscriber": "0"
   }
}
```

### detectLang field
`detectLang` is the result from [langdetect 1.0.7](https://pypi.python.org/pypi/langdetect?), 'NA' if no result returns.

### topicDetails field
`topicIds` and `relevantTopicIds` are resolved to entity name via the latest [Freebase data dump](https://developers.google.com/freebase/).
