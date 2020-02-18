import urllib as url

url = url.quote_plus('https://twitter.com/cambrasine/status/1196891901024243712')

handler = url.urlopen('https://api.proxycrawl.com/?token=Ylza1jJBRZ_JJxiqAUtnww&scraper=twitter-tweet&url=' + url)

print(handler.read())