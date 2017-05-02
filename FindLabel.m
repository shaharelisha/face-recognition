function [match_found, person] = FindLabel(j, k, WB, roi2)
% Given a binarized image WB, a matrix of blob location data roi2, and the indexes
% for specific blobs j and k, the area surrounding both blobs will be
% cropped out of WB, and fed into the OCR for text detection. The function
% will return whether a match was found, and if so, the label that was 
% detected. 

    width_total = roi2(j,3) + roi2(k,3);
    % region is expanded slightly in order to ensure the number is wholly
    % there
    number_box = [min(roi2(j:k,1))-15, max(roi2(j:k,2))-10, width_total+30, max(roi2(j:k,4))+20];
    cropped = imcrop(WB, number_box);
    
    % pass cropped image into ocr function, limiting the recognition to 
    % only accept digits and a single word.
    results = ocr(cropped, 'CharacterSet', '0123456789', 'TextLayout', 'Word');
    
    % if results found, it must be in the format digit-digit
    regularExpr = '\d\d';
    person = regexp(results.Text, regularExpr, 'match');
    try
        person = person{1};
        match_found = true;
    catch
        match_found = false;
        person = '0';
    end
end

