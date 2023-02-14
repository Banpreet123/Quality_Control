import math
class TaggerClassifier:
    """
    Class containing methods to tag a classifier as reliable/unreliable
    """
    def intervalLogs(self, tags) -> int:
        """
        Gets the time difference in seconds between subsequent tags, applies log base 2 to the result, 
        and finally averages the result to return it
        Args:
            tags (list): list of answer tags

        Returns:
            int: interval logs value for a tagger
        """
        if len(tags)==1:
            return -1                                       # Need to be discussed
        
        tags.sort(key = lambda l: l.created_at)             # sorting the tags based on created_at
        result = 0
        
        # Adding the log of time difference in seconds
        for ind in range(1, len(tags)):
            curr = (tags[ind].created_at - tags[ind-1].created_at).total_seconds()
            result = math.log(curr,2) if curr>0 else 0              # if time difference is 0 then add 0
        
        # Taking average
        result = result//(len(tags)-1)
        return result
        
        
    
    
    